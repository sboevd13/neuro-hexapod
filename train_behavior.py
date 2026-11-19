"""Train a small high-level neural utility model on simulated action outcomes.

The training target is the measured one-step reward after applying an action
in randomized planar obstacle scenes. This is a contextual-bandit baseline:
there is no long-horizon RL, online hardware adaptation or camera perception.
The navigation model is intentionally separate from the 18-joint motor PPO.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from autonomy.behavior import BehaviorModel, init_weights, state_features
from autonomy.explorer import SCAN_ANGLES, Rectangle, range_scan

# Simulated planar step, deliberately conservative compared with real robot.
STEP_DISTANCE = 0.10
TURN_ANGLE = 0.55
RADIUS = 0.21


def scenario(rng):
    """Random boxes and walls, with valid robot/goal position sampling."""
    boxes = [Rectangle(1.3, 0, .05, 1.35), Rectangle(-1.3, 0, .05, 1.35),
             Rectangle(0, 1.3, 1.35, .05), Rectangle(0, -1.3, 1.35, .05)]
    for _ in range(int(rng.integers(2, 8))):
        xy = rng.uniform(-.90, .90, 2)
        s = rng.uniform(.06, .16, 2)
        boxes.append(Rectangle(*xy, *s))
    return boxes


def clearance(xy, boxes):
    """Smallest distance from a point to a rectangular obstacle."""
    return min(float(np.linalg.norm(np.maximum(np.abs(np.asarray(xy) - [box.x, box.y]) -
                                             [box.half_x, box.half_y], 0.0))) for box in boxes)


def sample_state(rng, boxes):
    for _ in range(1000):
        pos = rng.uniform(-1.02, 1.02, size=2)
        if clearance(pos, boxes) > RADIUS + 0.035:
            break
    else:
        raise RuntimeError('No clear spawn')
    goal = rng.uniform(-1.02, 1.02, size=2)
    yaw = rng.uniform(-math.pi, math.pi)
    ranges, hits = range_scan(pos, yaw, boxes)
    return pos, yaw, goal, ranges


def action_rewards(pos, yaw, goal, boxes):
    """Reward is based on observed next-step progress and collisions.

    All three actions are counterfactually rolled out in the *training* 2D
    simulator. No obstacle positions are passed to the deployed policy.
    """
    before = float(np.linalg.norm(goal-pos))
    heading_error = math.atan2(goal[1]-pos[1], goal[0]-pos[0]) - yaw
    r = []
    for action in range(3):
        direction = np.array([math.cos(yaw), math.sin(yaw)])
        next_pos = pos + STEP_DISTANCE*direction if action == 0 else pos.copy()
        next_yaw = yaw + (TURN_ANGLE if action == 1 else (-TURN_ANGLE if action == 2 else 0))
        error = math.atan2(goal[1]-next_pos[1], goal[0]-next_pos[0])-next_yaw
        heading_gain = math.cos(error)-math.cos(heading_error)
        progress = before-np.linalg.norm(goal-next_pos)
        reward = 5.0*progress + 0.32*heading_gain - 0.01
        if action == 0:
            reward += 0.04
            # Swept body collision, not just the endpoint.
            for s in np.linspace(0.25, 1., 4):
                if clearance(pos + STEP_DISTANCE * s * direction, boxes) < RADIUS:
                    reward -= 3.0
                    break
        r.append(reward)
    return np.array(r, dtype=np.float32)


def data_set(n, seed):
    rng = np.random.default_rng(seed)
    x, y = [], []
    boxes = scenario(rng)
    for index in range(n):
        if index % 80 == 0:
            boxes = scenario(rng)
        pos, yaw, goal, ranges = sample_state(rng, boxes)
        x.append(state_features(ranges, yaw, pos, goal))
        y.append(action_rewards(pos, yaw, goal, boxes))
    return np.array(x, np.float32), np.array(y, np.float32)


def learn(weights, train_x, train_y, epochs=30, batch_size=256, lr=0.003, seed=13):
    """Full-information contextual-bandit regression with Adam + Huber loss."""
    rng = np.random.default_rng(seed)
    velocity = {k: np.zeros_like(v) for k, v in weights.items()}
    second = {k: np.zeros_like(v) for k, v in weights.items()}
    updates = 0
    for _ in range(epochs):
        order = rng.permutation(len(train_x))
        for s in range(0, len(order), batch_size):
            x, targets = train_x[order[s:s+batch_size]], train_y[order[s:s+batch_size]]
            h1 = np.tanh(x@weights['w1']+weights['b1'])
            h2 = np.tanh(h1@weights['w2']+weights['b2'])
            output = h2@weights['w3']+weights['b3']
            error = output-targets
            # Huber avoids exploding weights from rare collision penalties.
            grad = np.clip(error, -1, 1)/error.size
            g3 = h2.T@grad
            gb3 = grad.sum(axis=0)
            gh2 = (grad@weights['w3'].T)*(1-h2*h2)
            g2 = h1.T@gh2
            gb2 = gh2.sum(axis=0)
            gh1 = (gh2@weights['w2'].T)*(1-h1*h1)
            grads = {'w1': x.T@gh1, 'b1': gh1.sum(axis=0),
                     'w2': g2, 'b2': gb2, 'w3': g3, 'b3': gb3}
            updates += 1
            for key, g in grads.items():
                velocity[key] = 0.9*velocity[key]+0.1*g
                second[key] = 0.999*second[key]+0.001*g*g
                m = velocity[key]/(1-0.9**updates)
                v = second[key]/(1-0.999**updates)
                weights[key] -= lr*m/(np.sqrt(v)+1e-8)
    return weights


def metrics(model, x, y):
    prediction = model.values(x)
    best = np.argmax(y, axis=1)
    chosen = np.argmax(prediction, axis=1)
    regret = y[np.arange(len(y)), best]-y[np.arange(len(y)), chosen]
    return dict(samples=len(y), exact_action_accuracy=round(float(np.mean(chosen==best)), 4),
                mean_regret=round(float(np.mean(regret)), 5),
                optimal_reward=round(float(np.mean(y[np.arange(len(y)), best])), 5),
                chosen_reward=round(float(np.mean(y[np.arange(len(y)), chosen])), 5),
                collision_action_rate=round(float(np.mean((y[np.arange(len(y)), chosen] < -1.0))), 5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-samples', type=int, default=12000)
    p.add_argument('--test-samples', type=int, default=3000)
    p.add_argument('--epochs', type=int, default=32)
    p.add_argument('--output', type=Path, default=Path('models/behavior_v1.json'))
    p.add_argument('--report', type=Path, default=Path('runs/behavior_validation.json'))
    args = p.parse_args()
    if min(args.train_samples, args.test_samples, args.epochs) < 1:
        p.error('sample counts and epochs must be positive')
    training_x, training_rewards = data_set(args.train_samples, 13579)
    test_x, test_rewards = data_set(args.test_samples, 24680)
    before = BehaviorModel(init_weights())
    baseline_metrics = metrics(before, test_x, test_rewards)
    weights = learn(init_weights(), training_x, training_rewards, epochs=args.epochs)
    model = BehaviorModel(weights)
    after = metrics(model, test_x, test_rewards)
    report = {'model': 'contextual_one_step_action_value', 'training_seed': 13579,
              'held_out_seed': 24680, 'train_examples': args.train_samples,
              'eval_examples': args.test_samples, 'epochs': args.epochs,
              'before': baseline_metrics, 'after': after}
    model.save(args.output, report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
