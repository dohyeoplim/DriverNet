import pandas as pd

def create_submission(predictions: list, submission_path: str) -> None:
    num_classes = len(predictions[0])
    submission_df = pd.DataFrame(predictions, columns=[f'c{i}' for i in range(num_classes)])
    submission_df.insert(0, 'img', [f'img_{i}.jpg' for i in range(len(predictions))])
    submission_df.to_csv(submission_path, index=False)
