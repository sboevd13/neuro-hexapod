"""Compare information-seeking policies on REAL MuJoCo interaction outcomes.

Each candidate receives a low-cost short physical probe. Its expensive second
push is executed in a SHADOW physics run to obtain a counterfactual label.
The acquisition policy can inspect only the short-probe sensor vector and past
revealed outcomes. No test labels, mass, friction or mobility IDs enter it.
Thus this is a *pool-based sample efficiency benchmark*, not a claim that the
same saving in experimental energy has been achieved on the physical robot.
"""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from autonomy.active_experiments import (ACTION_DURATIONS, OutcomeEnsemble,
                                          acquire, dataset, regression_metrics,
                                          sanitize)


def physical_pool(start, count, output):
    """Generate randomized physics, with action duration independent of object."""
    from affordance_rollouts import perform
    rng = np.random.default_rng(start + 90133)
    order = rng.permutation(count)
    durations = [None] * count
    for i, ix in enumerate(order):
        durations[int(ix)] = ACTION_DURATIONS[i % len(ACTION_DURATIONS)]
    results = []
    for idx in range(count):
        seed = start + idx
        seconds = durations[idx]
        result = perform(seed, contact_long=seconds)
        result['second_push_seconds'] = seconds
        results.append(result)
        if idx % 32 == 0:
            print(f'physical {idx + 1}/{count}; valid={sum(r.get("valid",False) and not r.get("fell") for r in results)}',flush=True)
    output.mkdir(parents=True, exist_ok=True)
    # Raw data contains privileged simulator audit values; store for auditing only.
    (output / 'audit_full.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    clean = sanitize(results)
    (output / 'sensor_labels.json').write_text(json.dumps(clean,indent=2,allow_nan=False)+'\n')
    if len(clean) < .85 * count:
        raise RuntimeError('Too many invalid real-physics trials: cannot assess active learning')
    return clean


def compare_strategy(pool_x, pool_y, test_x, test_y, budget, strategy, repeat,
                     ensemble_members=4, steps=65):
    """Each strategy receives the SAME initial 8 revealed labels per repeat."""
    rng = np.random.default_rng(91013 + repeat)
    chosen = rng.permutation(len(pool_x))[:8].tolist()
    history = []
    for b in budget:
        while len(chosen) < b:
            model = OutcomeEnsemble.fit(pool_x[chosen],pool_y[chosen],
                      seed=repeat + 13, members=ensemble_members,steps=steps)
            rest = np.asarray([j for j in range(len(pool_x)) if j not in set(chosen)],dtype=int)
            count = min(b-len(chosen),len(rest))
            indices = acquire(strategy,model,pool_x[chosen],pool_x[rest],count,rng)
            chosen.extend(rest[indices].tolist())
        model = OutcomeEnsemble.fit(pool_x[chosen],pool_y[chosen],
                                   seed=repeat + 13,members=ensemble_members,steps=steps)
        metrics = regression_metrics(model,test_x,test_y,baseline=float(pool_y[chosen].mean()))
        history.append({'budget':b,'strategy':strategy,'repeat':repeat,'metrics':metrics})
    return history, chosen, model


def from_existing(train, test, budget, repeat_ids=(17,29,43), members=4, steps=65):
    """A deterministic fairness check that doesn't need MuJoCo for unit tests."""
    pool_x,pool_y=dataset(train)
    test_x,test_y=dataset(test)
    if len({tuple(v) for v in pool_x}) < budget[-1]:
        raise ValueError('Pool lacks enough distinct observable experiments')
    if budget[0] < 8 or budget[-1] > len(pool_x) or sorted(set(budget)) != list(budget):
        raise ValueError('Invalid label budget')
    if len(test_x)<8:
        raise ValueError('Insufficient test observations')
    results = []
    final_models = {}
    selected = {}
    for repeat in repeat_ids:
        for strategy in ('random','heuristic','uncertainty'):
            curves,ids,trained=compare_strategy(pool_x,pool_y,test_x,test_y,budget,strategy,repeat,members,steps)
            results.extend(curves)
            if repeat == repeat_ids[0] and strategy == 'uncertainty':
                final_models['uncertainty'] = trained
                selected['uncertainty'] = ids
    aggregate=[]
    for b in budget:
        for strategy in ('random','heuristic','uncertainty'):
            matching = [r for r in results if r['budget']==b and r['strategy']==strategy]
            aggregate.append({'budget':b,'strategy':strategy,'repeats':len(matching),
                              'mean_mae_m':round(statistics.mean(r['metrics']['mae_m'] for r in matching),6),
                              'mean_rmse_m':round(statistics.mean(r['metrics']['rmse_m'] for r in matching),6),
                              'min_mae_m':round(min(r['metrics']['mae_m'] for r in matching),6),
                              'max_mae_m':round(max(r['metrics']['mae_m'] for r in matching),6)})
    return {'version':1,'pool_candidates':len(train),'independent_test_examples':len(test),
            'budget':list(budget),'individual_curves':results,
            'aggregate':aggregate,'cost_metric':'number of labeled second-stage pushes',
            'disclosure':'All second pushes executed in shadow physics. Selection uses only sensor inputs and acquired labels, not simulator properties or future labels.'},final_models,selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool-start',type=int,default=3000)
    p.add_argument('--pool-size',type=int,default=200)
    p.add_argument('--test-start',type=int,default=6000)
    p.add_argument('--test-size',type=int,default=60)
    p.add_argument('--budgets',type=int,nargs='+',default=[8,16,32,64])
    p.add_argument('--members',type=int,default=4)
    p.add_argument('--steps',type=int,default=65)
    p.add_argument('--output',type=Path,default=Path('runs/active_learning'))
    p.add_argument('--reuse-physics',action='store_true',help='Read previously recorded physical trials without rerunning physics')
    args=p.parse_args()
    if (args.pool_size < max(args.budgets) or args.test_size < 8 or
            args.pool_start < 0 or args.test_start < 0 or
            set(range(args.pool_start,args.pool_start+args.pool_size)) &
            set(range(args.test_start,args.test_start+args.test_size)) or
            args.members < 3 or args.steps < 1):
        p.error('Invalid/disjoint pool and held-out sizes or training arguments')
    pool_path = args.output/'pool'
    test_path = args.output/'held_out'
    if args.reuse_physics:
        train=json.loads((pool_path/'sensor_labels.json').read_text())
        test=json.loads((test_path/'sensor_labels.json').read_text())
    else:
        train=physical_pool(args.pool_start,args.pool_size,pool_path)
        test=physical_pool(args.test_start,args.test_size,test_path)
    report,models,_=from_existing(train,test,args.budgets,members=args.members,steps=args.steps)
    report.update({'train_seeds':{'start':args.pool_start,'size':args.pool_size},
                   'test_seeds':{'start':args.test_start,'size':args.test_size},
                   'physics':'MuJoCo real contact with static/dynamic props',
                   'label':'signed total push displacement (m), sensor estimated',
                   'actions_seconds':list(ACTION_DURATIONS)})
    args.output.mkdir(parents=True,exist_ok=True)
    models['uncertainty'].save(args.output/'active_budget_model.json')
    # For the autonomous world, train a separate model on ALL available pool
    # labels; never on test scenes. Budget curves remain unaltered.
    full_x,full_y=dataset(train)
    held_x,held_y=dataset(test)
    full=OutcomeEnsemble.fit(full_x,full_y,seed=71,members=args.members,steps=max(90,args.steps))
    full.save(args.output/'active_world_model.json')
    report['full_pool_metrics']=regression_metrics(full,held_x,held_y,baseline=float(full_y.mean()))
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Only one carefully bounded best-effort diagnostic, not a claim of improvement.
    print('ACTIVE_LEARNING',json.dumps({'pool_size':len(train),'held_out_size':len(test),
                     'aggregate':report['aggregate']},allow_nan=False),flush=True)


if __name__=='__main__':main()
