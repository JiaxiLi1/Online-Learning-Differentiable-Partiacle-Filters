import subprocess
import concurrent.futures
import os
from itertools import product

train_types = ['DPF']
labeled_ratios = [1.0, 0.5, 0.01]
base_command = "python main_DiskTracking.py --gpu --dataset house3d -c ./configs/train.conf --resampler_type soft  --splitRatio 1  --num_epochs 30 --lr 1e-3 --lamda 0.01 --lr 1e-3 --std-x 0.04 --std-y 0.04 --std-t 5 --lamda 0.04"
parameters = list(product(train_types, labeled_ratios))
total_tasks = len(parameters)
semaphore = concurrent.futures.ThreadPoolExecutor(max_workers=3)

def run_command(args):
    train_type, labeled_ratio = args
    command = f"{base_command} --trainType {train_type} --labeledRatio {labeled_ratio}"
    process = subprocess.Popen(command, shell=True)
    process.wait()

futures = {semaphore.submit(run_command, params) for params in parameters[:3]}
parameters = parameters[3:]

while futures:
    done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)    
    for _ in range(len(done)):
        if parameters:
            future = semaphore.submit(run_command, parameters.pop())
            futures.add(future)
    print(f"Remaining tasks: {len(futures) + len(parameters)}") 