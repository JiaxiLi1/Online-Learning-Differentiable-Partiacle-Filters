import subprocess
import concurrent.futures
import os
from itertools import product

train_types = ['SDPF','DPF']
maze_ids = ['nav01', 'nav02', 'nav03']
labeled_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
base_command = "python main_DiskTracking.py --gpu --dataset maze --lr 1e-3 --seed 5 --std-x 20.0 --std-y 20.0 --std-t 0.5 --measurement cos --resampler_type soft --num_epochs 100"
parameters = list(product(train_types, maze_ids, labeled_ratios))
total_tasks = len(parameters)
semaphore = concurrent.futures.ThreadPoolExecutor(max_workers=3)

def run_command(args):
    train_type, maze_id, labeled_ratio = args
    command = f"{base_command} --trainType {train_type} --mazeID {maze_id} --labeledRatio {labeled_ratio}"
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