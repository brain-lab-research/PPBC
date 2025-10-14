import subprocess
import os
import argparse

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
# Configuration parameters
FEDERATED_METHODS = [
    "ppbc"
]
BASE_PARAMS = ['federated_method=ppbc',
    'federated_params.communication_rounds=1',
    "training_params.batch_size=256",
    "federated_params.print_client_metrics=False",
]

EPOCH_METHODS = ['angle', 'gradient_norm', 'loss', 'angle']
ITER_METHODS = ['loss', 'loss', 'random', 'random']

for e_m, i_m in zip(EPOCH_METHODS, ITER_METHODS):
    params = [
    f'federated_method.epoch_method={e_m}',
    f'federated_method.iter_method={i_m}',
    'federated_method.iterations=1']

    cmd = ["nohup", "python", "train.py"] + BASE_PARAMS + params
    output_path = f'scaffold_{e_m}_{i_m}.txt'
    cmd_str = " ".join(cmd) + f" > {output_path}"
    # Print and execute
    print(f"Running setup: {e_m}+{i_m}", flush=True)
    subprocess.run(cmd_str, shell=True, check=True)
