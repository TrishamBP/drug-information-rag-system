import os
import pandas as pd

DATA_DIR = "data/raw"

def load_txt_files():
    """Loads all .txt files from data/raw folder"""
    tables = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            name = filename.replace(".txt", "")
            path = os.path.join(DATA_DIR, filename)
            try:
                # Try tab delimiter first (most common for FDA files)
                print(f"Attempting to load {filename}...")
                try:
                    df = pd.read_csv(path, delimiter='\t', dtype=str, encoding="latin1", on_bad_lines='skip')
                    print(f"Tab delimiter successful for {filename}: {df.shape}")
                    if df.shape[1] == 1:  # If only 1 column, might be pipe-delimited
                        raise ValueError("Single column detected, trying pipe delimiter")
                except Exception as tab_error:
                    print(f"Tab delimiter failed for {filename}: {tab_error}")
                    # Fallback to pipe delimiter
                    try:
                        df = pd.read_csv(path, delimiter='|', dtype=str, encoding="latin1", on_bad_lines='skip')
                        print(f"Pipe delimiter successful for {filename}: {df.shape}")
                    except Exception as pipe_error:
                        print(f"Pipe delimiter also failed for {filename}: {pipe_error}")
                        raise pipe_error
                
                tables[name] = df
                print(f"✅ Successfully loaded {name}: {df.shape}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                # Continue with other files
    return tables