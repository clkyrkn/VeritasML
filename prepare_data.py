import pandas as pd
import os

def prepare_data():
    fake_path = "data/raw/Fake.csv"
    true_path = "data/raw/True.csv"
    output_path = "data/raw/news.csv"

    # Read files
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # add 'label' column
    fake_df["label"] = 1
    true_df["label"] = 0

    # Use only ‘text’ and ‘label’ columns
    fake_df = fake_df[["text", "label"]]
    true_df = true_df[["text", "label"]]

    # Combine and mix
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save
    os.makedirs("data/raw", exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"Data prepared: {output_path}")

if __name__ == "__main__":
    prepare_data()