import os
import re
import pandas as pd

def merge_nirf_files():
    base_dir = "../../data/kaggle/nirf/nirf_rankings_2016_2025"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, base_dir))
    
    # ensure output directory exists
    output_dir = os.path.abspath(os.path.join(script_dir, "../../data/processed"))
    os.makedirs(output_dir, exist_ok=True)
    
    all_dfs = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and "NIRF" in filename:
            # Extract year from filename
            match = re.search(r"20\d{2}", filename)
            if not match:
                continue
            
            year = int(match.group())
            filepath = os.path.join(data_dir, filename)
            
            # Read CSV
            try:
                # Some files might have different encodings
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin-1')
                
            # Add Year column
            df['Year'] = year
            
            # Print columns to see if sub-scores exist
            print(f"[{year}] columns: {list(df.columns)}")
            
            all_dfs.append(df)
            
    # Concatenate all DataFrames
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Check for subscores in the master
    subscores = ['TLR', 'RPC', 'GO', 'OI', 'PERCEPTION']
    found_subscores = [col for col in subscores if col in master_df.columns]
    
    print("\n--- Summary ---")
    print(f"Total rows in merged dataset: {len(master_df)}")
    print(f"Columns in merged dataset: {list(master_df.columns)}")
    print(f"Sub-scores found historically: {found_subscores}")
    
    # Optional: If the columns are completely different across years, print which years have them
    for score in subscores:
        years_with_score = sorted(master_df[master_df[score].notna()]['Year'].unique())
        print(f"  {score} present in years: {years_with_score}")
        
    output_path = os.path.join(output_dir, "NIRF_Master_2016_2025.csv")
    master_df.to_csv(output_path, index=False)
    print(f"\nMaster file saved to: {output_path}")

if __name__ == "__main__":
    merge_nirf_files()
