import os
import json
import glob
import pandas as pd
import numpy as np

def process_results(file_pattern):
    records = []
    
    # Load all matching JSON result files
    for file in glob.glob(file_pattern):
        with open(file, 'r') as f:
            data = json.load(f)
            for run in data:
                dataset = run["dataset"]
                method = run["method"]
                seed = run["seed"]
                
                # Handle varying JSON structures between forest methods and foundation model
                if method == "foundation":
                    acc = run["result"]["metrics"]["balanced_accuracy"]
                    compute = run["result"]["elapsed_sec"]
                else:
                    acc = run["final_result"]["metrics"]["balanced_accuracy"]
                    # Total compute = search phase + final refit/test evaluation phase
                    compute = run["search_seconds"] + run["final_result"]["elapsed_sec"]
                    
                records.append({
                    "Dataset": dataset,
                    "Method": method.capitalize(),
                    "Seed": seed,
                    "Accuracy": acc,
                    "Compute (s)": compute
                })
                
    df = pd.DataFrame(records)
    
    if df.empty:
        return df

    # Enforce strictly increasing row count order for the final tables
    dataset_order = ["breast-w", "credit-g", "phoneme", "electricity", "covertype"]
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    
    return df

def display_terminal_tables(df):
    # Calculate mean and standard deviation across seeds
    agg_df = df.groupby(["Dataset", "Method"], observed=False).agg(
        Acc_mean=("Accuracy", "mean"),
        Acc_std=("Accuracy", "std"),
        Comp_mean=("Compute (s)", "mean"),
        Comp_std=("Compute (s)", "std")
    ).reset_index()
    
    # Format as "Mean ± Std" strings
    agg_df["Balanced Accuracy"] = agg_df.apply(
        lambda r: f"{r['Acc_mean']:.3f} ± {r['Acc_std']:.3f}" if pd.notna(r['Acc_mean']) else "-", axis=1
    )
    agg_df["Compute Time (s)"] = agg_df.apply(
        lambda r: f"{r['Comp_mean']:.1f} ± {r['Comp_std']:.1f}" if pd.notna(r['Comp_mean']) else "-", axis=1
    )
    
    # Pivot to create Method columns with Dataset rows
    acc_pivot = agg_df.pivot(index="Dataset", columns="Method", values="Balanced Accuracy")
    comp_pivot = agg_df.pivot(index="Dataset", columns="Method", values="Compute Time (s)")
    
    # Order columns consistently 
    method_order = ["Default", "Random", "Smbo", "Hyperband", "Foundation"]
    valid_cols = [m for m in method_order if m in acc_pivot.columns]
    acc_pivot = acc_pivot[valid_cols]
    comp_pivot = comp_pivot[valid_cols]
    
    # Configure pandas to print cleanly in the terminal without wrapping
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n=== Predictive Performance (Balanced Accuracy) ===")
    print(acc_pivot.fillna("-"))
    
    print("\n=== Computational Effort (Seconds) ===")
    print(comp_pivot.fillna("-"))
    print("\n")

if __name__ == "__main__":
    # Join the directory path and filter exclusively for the 'course' profile runs
    target_dir = "run_exp_results"
    search_pattern = os.path.join(target_dir, "results_course_*.json")
    
    raw_results = process_results(search_pattern)
    
    if not raw_results.empty:
        display_terminal_tables(raw_results)
    else:
        print(f"No JSON files found matching pattern: {search_pattern}")