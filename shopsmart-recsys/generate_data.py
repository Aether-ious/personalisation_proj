# generate_data.py
import pandas as pd
import numpy as np
import os

def generate_dummy_data(num_rows=10000, output_path="data/raw/interactions_train.csv"):
    print(f"Generating {num_rows} rows of dummy data...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Random Users (0 to 9999) and Items (0 to 4999)
    user_ids = np.random.randint(0, 10000, size=num_rows)
    item_ids = np.random.randint(0, 5000, size=num_rows)
    
    # Random targets: 1 = Click, 0 = No Click
    # We weight it so 30% are clicks (imbalanced classes are common in RecSys)
    targets = np.random.choice([0, 1], size=num_rows, p=[0.7, 0.3])
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "target": targets
    })
    
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    generate_dummy_data()