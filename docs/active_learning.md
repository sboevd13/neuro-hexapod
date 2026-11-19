# Active experiment selection in MuJoCo — research prototype

This is an opt-in extension of `feature/embodied-learning-v1`, **not** a deployment system. It measures how useful different strategies are for selecting physical object-push experiments. Servo gait V5 / optional PPO locomotion remains unchanged.

## What actually learns

Before the second physical push, the robot has a simulated 23-ray range scan and the result of a short physical contact. The **only** permitted input features for regression are five already normalized sensor-derived observations (apparent width, range, short-probe displacement, contact, observation count) and the **duration of the second action** (0.24 or 0.55 s). The new regressor predicts the signed total object-surface displacement after the push, measured in metres. A bootstrap ensemble of four small neural regressors supplies a mean forecast and an uncalibrated *model disagreement* estimate.

Simulator mass, friction, whether a body is freejoint/static, object IDs and exact object motion are never passed into training or selection. The ground-truth physical properties are recorded only in separate **audit** files.

## Experiment selection and fair comparison

In the pool-based experiment, MuJoCo simulates 200 candidate objects, each with its own independently sampled size, mass, friction and mobility. A preliminary short probe is available on every candidate. A longer push (randomized 0.24 / 0.55 seconds independently of the object's properties) is also executed in a **shadow physics** rollout for each object so that its unobserved outcome can be evaluated later.

At a budget of 8, 16, 32 and 64 revealed **second-push outcomes**, three acquisition strategies see **exactly the same** eight initial labeled examples per repeat:

- Random: request arbitrary previously hidden labels.
- Hand-coded heuristic: select objects with larger observed initial motion and apparent geometry novelty.
- Epistemic strategy: choose larger disagreement among bootstrap ensemble predictions, adjusted for diversity in observed sensor features.

No strategy is allowed to read the remaining labels or simulator state when choosing. This is a **counterfactual label-acquisition benchmark**: all shadow second pushes really were simulated to measure what would have happened, so it does **not** demonstrate energy or time saved in a deployed robot. The 60 additional scenes are disjoint from the 200 candidates and never used to select or fit models.

MAE and RMSE of signed object displacement in metres are reported for each label budget and acquisition strategy, across three acquisition seeds. The successful result must be **measured**; it is not assumed that epistemic selection beats random or that any model has useful calibrated uncertainty.

## Reproduce the physical benchmark

Activate the repository's Python environment with `mujoco==3.2.7` and NumPy:

```bash
python -m unittest discover -s tests -p test_active_experiments.py -v
python active_learning_benchmark.py --pool-start 3000 --pool-size 200 \
  --test-start 6000 --test-size 60 --budgets 8 16 32 64 \
  --output runs/active_learning
```

This produces `runs/active_learning/report.json`, a 4-model `active_budget_model.json`, an independently trained full-pool `active_world_model.json`, the underlying sensor-only labeled examples and separate privileged physical-audit files. Reuse previously simulated scenes with `--reuse-physics` to regenerate selection curves without rerunning physics. Original evaluation is automated in `.github/workflows/active-learning.yml`.

## Opt-in continuous-world trial

```bash
python autonomous_world.py --seed 11 --seconds 90 \
  --model models/affordance_outcome_experimental.json \
  --experience models/affordance_training_experiences.json \
  --experiment-model runs/active_learning/active_world_model.json \
  --experiment-experience runs/active_learning/pool/sensor_labels.json \
  --exploration-weight 2.0 --output runs/active_world --view
```

In a single MuJoCo room the existing frontier explorer chooses targets. After contact and a short probe, the new ensemble **optionally** decides between stopping (`observe`), a gentle additional push (0.24 s), or a longer additional push (0.55 s). The robot then measures surface motion. **Only fully executed second pushes with observed outcomes** are labeled and appended to its replay. A high-disagreement observation may trigger a gentle second push even if the predicted motion is small; this exploratory override is restricted to at most two extra pushes per scene and is disabled by the height safety veto. With enough examples, the ensemble retrains during the same episode, and saves the updated `active_predictor.json` and `active_experiences.json`. Deliberately declining a second push is not mislabeled as “immobile.” The action selector is a rule-based utility function using neural forecasts and disagreement; it is **not a learned full robot policy**.

Safety: the upper layers veto actions on low body height, use bounded durations, and preserve the original V5 gait, but there are **no actual motor torque limits or verified hardware collision bounds**. Contact is intentional **only in the simulator**. Do not run these commands on a physical hexapod without proper force sensing, current limits, watchdog and emergency stop.

## Limitations

- Only planar idealized range rays and MuJoCo ground-truth robot pose; no RGB recognition, SLAM, reliable object tracking, or real tactile feedback.
- The model's disagreement is a heuristic and is **not calibrated predictive uncertainty**.
- Pre-probing *all* candidates in the benchmark incurs real simulated work even when their second-stage labels are hidden. Thus sample-efficiency curves show **second-push label efficiency**, not overall sensor cost or physical battery savings.
- A small network and 260 randomized procedural examples cannot establish universal transfer to unfamiliar objects, landscapes or real hardware.
- Regression improvement and benefit of active label acquisition are separate questions; results may show no superiority over random strategies.
- Original navigation and locomotion stay the defaults; the experimental features require explicit CLI flags and branch selection.
