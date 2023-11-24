import subprocess
import concurrent.futures
import os
import signal
from itertools import product

# Define parameter options
train_types = ['SDPF', 'DPF']
maze_ids = ['nav01', 'nav02', 'nav03']
labeled_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
# train_types = ['SDPF', 'DPF']
# maze_ids = ['nav01', 'nav02', 'nav03']
# labeled_ratios = [1.0, 0.5]

base_command = "python main_DiskTracking.py --gpu --dataset maze --lr 1e-3 --seed 5 --std-x 20.0 --std-y 20.0 --std-t 0.5 --measurement cos --resampler_type soft --num_epochs 100"


# Create combinations of parameters
parameters = list(product(train_types, maze_ids, labeled_ratios))
total_tasks = len(parameters)
print(f"Total tasks to run: {total_tasks}")

# Use a semaphore to limit the number of running tasks to 7
semaphore = concurrent.futures.ThreadPoolExecutor(max_workers=7)
pids = []

def run_command(args):
    train_type, maze_id, labeled_ratio = args
    command = f"{base_command} --trainType {train_type} --mazeID {maze_id} --labeledRatio {labeled_ratio}"
    process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    pids.append(process.pid)  # store the pid
    process.wait()

try:
    # Submit the first 7 tasks to the executor
    futures = {semaphore.submit(run_command, params) for params in parameters[:7]}
    parameters = parameters[7:]

    while futures:
        # Wait for the first task to complete
        done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
        
        # Submit a new task for each completed task
        for _ in range(len(done)):
            if parameters:
                future = semaphore.submit(run_command, parameters.pop())
                futures.add(future)
        
        print(f"Remaining tasks: {len(futures) + len(parameters)}")
    
except KeyboardInterrupt:
    print("Cancelling running tasks...")
    # kill all the subprocesses
    for pid in pids:
        os.killpg(os.getpgid(pid), signal.SIGTERM)  # send SIGTERM to the process group
    print("Cancelled remaining futures")


