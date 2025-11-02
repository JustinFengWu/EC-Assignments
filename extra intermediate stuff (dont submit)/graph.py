import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder names for each problem
folders = [
    "split_runs_f2101", "split_runs_f2102", "split_runs_f2103", "split_runs_f2104",
    "split_runs_f2201", "split_runs_f2202", "split_runs_f2203", "split_runs_f2204",
    "split_runs_f2301", "split_runs_f2302", "split_runs_f2303"
]

# Output folder for plots
os.makedirs("tradeoff_plots1", exist_ok=True)

# For each problem folder
for folder in folders:
    problem_id = folder.split("_f")[-1]  # e.g., "2101"
    first_run_file = os.path.join(folder, f"f{problem_id}_run-1.csv")

    # Skip if missing
    if not os.path.exists(first_run_file):
        print(f"⚠️ Missing: {first_run_file}")
        continue

    # Load first run
    df = pd.read_csv(first_run_file)

    # --- Construct artificial "cost" axis ---
    # Use evaluation count as proxy for cost
    df["cost"] = df["evaluations"]
    df["reward"] = df["raw_y"]

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(df["cost"], df["reward"], color="tab:red", lw=2)
    plt.xlabel("Cost (Evaluations)")
    plt.ylabel("Reward / f(x)")
    plt.title(f"Trade-off Curve (Problem {int(problem_id) - 1}) - Run 1")
    plt.grid(True, alpha=0.3)

    # Save plot
    out_path = f"tradeoff_plots1/tradeoff_f{int(problem_id) - 1}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"✅ Saved {out_path}")

print("\nAll trade-off plots generated and saved in 'tradeoff_plots' folder.")
