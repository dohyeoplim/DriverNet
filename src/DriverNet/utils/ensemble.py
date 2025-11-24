import pandas as pd
from typing import List

def average_predictions(submission_paths: List[str], output_path: str) -> None:
    if not submission_paths:
        print("No submission files to average.")
        return

    all_submissions = [pd.read_csv(path) for path in submission_paths]

    image_col = all_submissions[0].columns[0]

    ensembled_df = pd.concat(all_submissions).groupby(image_col).mean().reset_index()

    ensembled_df.to_csv(output_path, index=False)
    print(f"Ensembled submission saved to {output_path}")

