import os, csv, json, itertools, subprocess

qubit_list = [2,4,6,8]
layers_list = [1,3,5,7,10]
keys_list = [5,35,1031]

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
results_dir = os.path.join(root_dir,"results")
os.makedirs(results_dir,exist_ok=True)


base_name = "sweep_results"
csv_path = os.path.join(results_dir, f"{base_name}.csv")
if os.path.exists(csv_path):
    counter = 1
    while True:
        csv_path = os.path.join(results_dir, f"{base_name}({counter}).csv")
        if not os.path.exists(csv_path):
            break
        counter += 1
temp_json_path = os.path.join(results_dir, "temp_metrics.json")

with open(csv_path,mode="w",newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Qubits","Layers","Key","Loss","Time_Seconds","Status"])

    combinations = list(itertools.product(qubit_list,layers_list,keys_list))
    total_runs = len(combinations)

    print(f"Starting sweep with {total_runs} combinations...")
    print(f"Results will be saved to: {csv_path}\n")

    print("-" * 72)
    print(f"{'Run':<9}{'Qubits':<9}{'Layers':<9}{'Key':<9}{'Status':<13}{'Loss':<13}{'Time (s)':<10}")
    print("-" * 72)
    
    for i, (q,l,k) in enumerate(combinations,start=1):
        run_str = f"{i}/{total_runs}"
        print(f"{run_str:<9}{q:<9}{l:<9}{k:<9}{'Running...':<13}",
              end="",flush=True)

        cmd = ["python", "ode_optimization_optax.py",
               "--n_qubits", str(q),
               "--n_layers", str(l),
               "--key", str(k),
               "--output_json", temp_json_path]

        try:
            result = subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)

            if os.path.exists(temp_json_path):
                with open(temp_json_path, "r") as f:
                    metrics_data = json.load(f)

                final_loss = metrics_data["loss"]
                run_time = metrics_data["time"]
                status = "Success"

                os.remove(temp_json_path)
                print(f"\r{run_str:<9}{q:<9}{l:<9}{k:<9}{status:<13}{final_loss:<13.2e}{run_time:<10.1f}")
            else:
                final_loss = "N/A"
                run_time = "N/A"
                status = "Missing Output File"
                print(f"\r{run_str:<9}{q:<9}{l:<9}{k:<9}{status}")

        except subprocess.CalledProcessError as e:
            final_loss = "N/A"
            run_time = "N/A"
            status = f"Crashed: {e}"
            print(f"\r{run_str:<9}{q:<9}{l:<9}{k:<9}{status}")

        writer.writerow([q, l, k, final_loss, run_time, status])
        csv_file.flush()

print("\nSweep completed successfully!")
