import os
import numpy as np


def extract_and_process_data(root_dir):
    # Define possible parameters
    dpf_params = ["DPF", "SDPF_elbo", "SDPF_pl"]
    number_params = ['0.0', '0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1.0']
    # number_params = ['5e-05', '0.0001', '0.0005', '0.001', '0.005', '0.01', '0.05']

    # Create a dictionary to store the results with placeholder values
    results = {}
    for dpf in dpf_params:
        key = f"{dpf}"
        results[key] = [None] * len(number_params)

    # Define the path to the 'logs' folder within 'DPF-Merge'
    logs_dir = os.path.join(root_dir, "test")

    # Iterate over the folders inside 'logs' and process them
    for folder in os.listdir(logs_dir):
        folder_path = os.path.join(logs_dir, folder)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Check for each parameter set ||||||
            for dpf in dpf_params:
                for idx, num in enumerate(number_params):
                    if f"0.02_1_500_1_{dpf}_{num}" in folder:
                        # Extract the number parameter from the folder name
                        # Load the data from the test.npy file inside the 'data' folder
                        data_path = os.path.join(folder_path, "aesmc_parameter_error_recorder_0.02_1_500_1.npy")
                        if os.path.exists(data_path):
                            data = np.load(data_path)[0]
                            # Set the mean of the data to the results dictionary at the correct index
                            results[f"{dpf}"][idx] = data[-1]

    return results

if __name__ == "__main__":
    current_directory = os.getcwd()
    data_results = extract_and_process_data(current_directory)
    for key, value in data_results.items():
        print(key, ":", value)