import subprocess
import concurrent.futures
import os
from itertools import product
import sys
sys.path.append("/home/jiaxi/Insync/jl02764@surrey.ac.uk/OneDrive Biz/py_project/PycharmProjects/aesmc1")



train_types = ['SDPF_pl']
labeled_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]
base_command = "python test_losses.py --resampler_type normal --device cuda"
parameters = list(product(train_types, labeled_ratios))
total_tasks = len(parameters)
print("total run:",total_tasks)
semaphore = concurrent.futures.ThreadPoolExecutor(max_workers=7)

def run_command(args):
    train_type, labeled_ratio = args
    command = f"{base_command} --trainType {train_type} --labelled_ratio {labeled_ratio}"
    process = subprocess.Popen(command, shell=True)
    process.wait()

futures = {semaphore.submit(run_command, params) for params in parameters[:7]}
parameters = parameters[7:]

while futures:
    done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)    
    for _ in range(len(done)):
        if parameters:
            future = semaphore.submit(run_command, parameters.pop())
            futures.add(future)
    print(f"Remaining tasks: {len(futures) + len(parameters)}") 
