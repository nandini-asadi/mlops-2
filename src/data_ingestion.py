import pandas as pd
from sklearn.datasets import load_diabetes
def load_and_process_data():
    data = load_diabetes(as_frame=True)
    df = data.frame

    df.to_csv("../data/raw/diabetes.csv", index=False)

    df_processed = df.copy()

    df_processed.to_csv("../data/processed/diabetes_processed.csv", index=False)

    print("Data injestion completed")

if __name__ == "__main__":
    load_and_process_data()