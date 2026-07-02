import os
import subprocess
import itertools
import csv
import re

qubit_list = [2,4,6,8]
layers_list = [1,3,5,7,10]
keys_list = [5,35,1031]

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
results_dir = os.path.join(root_dir,"results")
os.makedirs(results_dir,exist_ok=True)

csv_path = os.path.join(results_dir, "sweep_results.csv")

with open(csv_path,mode="w",newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Qubits","Layers","Key","Loss","Time_Seconds","Status"])

    combinations = list(itertools.product(qubit_list,layers_list,keys_list))
    total_runs = len(combinations)

    print(f"Starting sweep with {total_runs} combinations...")
    print(f"Results will be saved to: {csv_path}\n")
    
    for i, (q,l,k) in enumerate(combinations,start=1):
        print(f"[{i}/{total_runs}] Running: Qubits={q}, Layers={l}, Key={k}...",
              end="",flush=True)
        
        cmd = [
            "python", "optjx2.py",
            "--n_qubits", str(q),
            "--n_layers", str(l),
            "--key", str(k)
        ]

        try:
            result = subprocess.run(cmd,capture_output=True,
                                    text=True,check=True)
            output = result.stdout

            # Use Regular Expressions to find the Loss and Time from your print statements
            loss_match = re.search(r"Loss:\s+([0-9\.eE+-]+)", output)
            time_match = re.search(r"took:\s+([0-9\.]+)\s+seconds", output)
            
            if loss_match and time_match:
                final_loss = float(loss_match.group(1))
                run_time = float(time_match.group(1))
                status = "Success"
                print(f"Done! Loss: {final_loss:.2e}, Time: {run_time:.1f}s")
            else:
                final_loss = "N/A"
                run_time = "N/A"
                status = "Output Parsing Failed"
                print("Failed to parse output!")
                
        except subprocess.CalledProcessError as e:
            # If the script crashes (e.g., out of memory), log the failure and keep going
            final_loss = "N/A"
            run_time = "N/A"
            status = f"Crashed: {e}"
            print("Crashed!")
            
        # Write the row to the CSV immediately so data isn't lost if you cancel midway
        writer.writerow([q, l, k, final_loss, run_time, status])
        csv_file.flush() # Force write to disk

print("\nSweep completed successfully!")
