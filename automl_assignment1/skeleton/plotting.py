import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_optimization_history(file_pattern):
    history_records = []
    foundation_records = []
    
    for file in glob.glob(file_pattern):
        with open(file, 'r') as f:
            data = json.load(f)
            for run in data:
                dataset = run["dataset"]
                method = run["method"]
                seed = run["seed"]
                
                if method == "foundation":
                    foundation_records.append({
                        "Dataset": dataset,
                        "Seed": seed,
                        "Compute (s)": run["result"]["elapsed_sec"],
                        "Accuracy": run["result"]["metrics"]["balanced_accuracy"]
                    })
                elif method in ["random", "smbo", "hyperband"]:
                    cumulative_time = 0.0
                    best_objective = -np.inf
                    
                    for step in run["history"]:
                        cumulative_time += step["elapsed_sec"]
                        # Enforce monotonic increase per run
                        if step["objective"] > best_objective:
                            best_objective = step["objective"]
                        
                        history_records.append({
                            "Dataset": dataset,
                            "Method": method.capitalize(),
                            "Seed": seed,
                            "Cumulative Time (s)": cumulative_time,
                            "Best Validation Accuracy": best_objective
                        })
                    
                    # Extend the final flat line to the end of the search phase
                    history_records.append({
                        "Dataset": dataset,
                        "Method": method.capitalize(),
                        "Seed": seed,
                        "Cumulative Time (s)": run["search_seconds"],
                        "Best Validation Accuracy": best_objective
                    })

    hist_df = pd.DataFrame(history_records)
    found_df = pd.DataFrame(foundation_records)
    return hist_df, found_df

def plot_optimization_progress(hist_df, found_df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    
    datasets = ["breast-w", "credit-g", "phoneme", "electricity", "covertype"]
    
    for dataset in datasets:
        dataset_hist = hist_df[hist_df["Dataset"] == dataset]
        if dataset_hist.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Plot each seed as an individual line to prevent chaotic interpolations
        sns.lineplot(
            data=dataset_hist,
            x="Cumulative Time (s)",
            y="Best Validation Accuracy",
            hue="Method",
            units="Seed",
            estimator=None,  # Disables the jumping mean calculation
            alpha=0.7,       # Adds slight transparency to overlapping runs
            drawstyle="steps-post",
            ax=ax,
            linewidth=1.5
        )
        
        # Overlay Foundation model if evaluating covertype
        if dataset == "covertype" and not found_df.empty:
            found_data = found_df[found_df["Dataset"] == "covertype"]
            mean_acc = found_data["Accuracy"].mean()
            mean_time = found_data["Compute (s)"].mean()
            
            ax.scatter(
                [mean_time], [mean_acc], 
                color="red", 
                s=100, 
                zorder=5, 
                label=f"Foundation (Mean)\nAcc: {mean_acc:.3f}, Time: {mean_time:.0f}s"
            )
            ax.axhline(mean_acc, color="red", linestyle="--", alpha=0.6, zorder=4)

        ax.set_title(f"Optimization Progress: {dataset}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Cumulative Search Time (Seconds)", fontsize=10)
        ax.set_ylabel("Validation Balanced Accuracy", fontsize=10)
        
        # Deduplicate the legend (estimator=None creates a label for every single line)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Method", fontsize=9, title_fontsize=10)
        
        fig.tight_layout()
        output_path = os.path.join(output_dir, f"opt_progress_{dataset}.png")
        # Save as PNG at high resolution
        fig.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Generated: {output_path}")

if __name__ == "__main__":
    target_dir = "run_exp_results"
    search_pattern = os.path.join(target_dir, "results_course_*.json")
    
    hist_df, found_df = extract_optimization_history(search_pattern)
    
    if not hist_df.empty:
        plot_optimization_progress(hist_df, found_df)
    else:
        print(f"No JSON files found matching pattern: {search_pattern}")