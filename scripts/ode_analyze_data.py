import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
csv_path = os.path.join(root_dir, "results", "sweep_results.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find results file ate: {csv_path}")

df = pd.read_csv(csv_path)
df_clean = df[df["Status"] == "Success"].copy()
df_clean["Loss"] = pd.to_numeric(df_clean["Loss"])

averages = df_clean.groupby(["Qubits","Layers"])["Loss"].mean().reset_index()
averages = averages.rename(columns={"Loss": "Average_Loss"})

print("--- Average Loss Table ---")
print(averages.to_string(index=False,
                         formatters={"Average_Loss": "{:.6e}".format}))

pivot_matrix = averages.pivot(index="Layers",
                              columns="Qubits",
                              values="Average_Loss")
print(pivot_matrix.map(lambda x: f"{x:.6e}"))
