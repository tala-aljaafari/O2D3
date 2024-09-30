# Supercharging OOD Detection

This repository contains the implementation for the first part of my thesis on Supercharging Out-of-Distribution (OOD) Detection.

## Structure

```
- **/src/**: Contains the main source code for the detectors and environments.
  - **/detectors/**: Implementation of DEXTER, RBFDEXTER, PEDM, and OCD.
  - **/envs_continuous/**: The environments we test on (Pusher, Reacher, HalfCheetah).
    - **anom_mj_env.py**: The code for generating cross-dimensionally correlated anomalies.

- **/assets/**: Contains the policies and rollouts needed for each experiment.

- **/CUSUM/**: Contains the CUSUM extension of RBFDEXTER.
```
## Example Commands
To test RBFDEXTER on Reacher, with observation noise strength of 0.4 (corresponding to Light Noise), 1-step noise correlation for testing, use the following command:

```
python train_test_detector_continuous_env.py \
    --detector-name "RBFDEXTER_Detector" \
    --train-env-id "MJReacher-v0" \
    --test-env-id "MJReacher-v0" \
    --train-data-path "../assets/rollouts/IMANOReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p4/50_ep/ep_data.pkl" \
    --policy-path "../assets/policies/IMANOReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p4/best_model.zip_jit.pt" \
    --train-env-noise-corr "(0.0,0.0)" \
    --test-env-noise-corr "(0.95,0.0)" \
    --train-noise-strength 0.4 \
    --test-noise-strength 0.4 \
    --noise-mode "obs" \
    --num-train-episodes 2000 \
    --num-test-episodes 100 \
    --experiment-id "2023_12_01_12_00_00" \
    --seed 2023
```

## Credits

This repository draws heavily from the following projects:

DEXTER (https://github.com/LinasNas/DEXTER.git) - for the core OOD detection framework.
illusory-attacks (https://github.com/LinasNas/illusory-attacks.git) - for the code used to train the agent policies.
