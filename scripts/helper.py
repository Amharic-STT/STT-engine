import pickle
import pandas as pd

def read_obj(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_obj(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_csv(csv_path, missing_values=[]):
    try:
        df = pd.read_csv(csv_path, na_values=missing_values)
        print("file read as csv")
        return df
    except FileNotFoundError:
        print("file not found")


def save_csv(df, csv_path):
    try:
        df.to_csv(csv_path, index=False)
        print('File Successfully Saved.!!!')

    except Exception:
        print("Save failed...")

    return df
