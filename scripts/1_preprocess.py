from datasets import load_dataset
import pandas as pd

# Step 1: Load the GEM Cochrane dataset (with custom loader)
print(" Loading dataset...")
dataset = load_dataset("GEM/cochrane-simplification", trust_remote_code=True)

# Step 2: Convert each split to pandas DataFrame and keep only 'source' and 'target'
def save_split(split_name):
    print(f" Processing split: {split_name}")
    df = dataset[split_name].to_pandas()[["source", "target"]]
    df.to_csv(f"data/cochrane_{split_name}.csv", index=False)
    print(f" Saved data/cochrane_{split_name}.csv")

# Step 3: Save train, validation, and test splits
save_split("train")
save_split("validation")
save_split("test")

