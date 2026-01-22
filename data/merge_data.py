# ./data/merge_data.py
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def merge_datasets():
    path = "datasets/preprocessed/"
    all_files = glob.glob(os.path.join(path, "clean_dataset_*.csv"))
    
    print(f"Found: {len(all_files)} files.")
    
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        df_list.append(df)
        print(f"   - {os.path.basename(filename)}: {len(df)} lines")

    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    
    print(f"Total rows: {len(full_df)} lines")
    print("="*40)

    label_counts = full_df['label'].value_counts().sort_index()
    print("\nSENTIMENT DISTRIBUTION:")
    print(label_counts)

    neg_count = label_counts.get(0, 0)
    pos_count = label_counts.get(2, 0)
    
    if pos_count > 5 * neg_count:
        print("\nWARNING: Positive samples are >5x Negative samples.")
        print("Trainer will automatically apply 'Class Weights'.")
    
    output_path = "datasets/merged_data.csv"
    full_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nMerged dataset saved at: {output_path}")

if __name__ == "__main__":
    merge_datasets()