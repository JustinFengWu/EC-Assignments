import os
import pandas as pd

# Base directory containing all problem folders
base_dir = "complete_logs\GSEMO"

# Problem IDs to process
problem_ids = [2201, 2202, 2203, 2204]

# Loop through each problem ID
for pid in problem_ids:
    # Build expected folder and file paths
    data_folder = os.path.join(base_dir, f"data_f{pid}_MaxCoverage{pid}") \
        if pid < 2200 else os.path.join(base_dir, f"data_f{pid}_MaxInfluence{pid}")
    dat_path = os.path.join(data_folder, f"IOHprofiler_f{pid}_DIM4039.dat")

    if not os.path.exists(dat_path):
        print(f"⚠️ Skipping {pid}: file not found at {dat_path}")
        continue

    # Create output folder for split runs
    output_dir = os.path.join(base_dir, f"split_runs_f{pid}")
    os.makedirs(output_dir, exist_ok=True)

    # Read the .dat file
    with open(dat_path, "r") as f:
        lines = f.readlines()

    runs = []
    current_run = []

    # Split by header "evaluations raw_y"
    for line in lines:
        if line.strip().startswith("evaluations raw_y"):
            if current_run:
                runs.append(current_run)
                current_run = []
        current_run.append(line.strip())
    if current_run:
        runs.append(current_run)

    # Save each run as a CSV
    for i, run_lines in enumerate(runs, start=1):
        # Extract only data lines (skip header)
        data = [l.split() for l in run_lines if not l.startswith("evaluations")]
        if not data:
            continue
        df = pd.DataFrame(data, columns=["evaluations", "raw_y"]).astype(float)

        # Output file path
        out_path = os.path.join(output_dir, f"f{pid}_run-{i}.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Saved {out_path} ({len(df)} evaluations)")

    print(f"🎯 Done splitting f{pid} into {len(runs)} runs.\n")

print("✨ All requested problem files have been processed successfully!")
