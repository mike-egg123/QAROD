import pandas as pd
import os

def get_filepaths(parent):
    filepaths = []
    for filepath, dirnames, filenames in os.walk(parent):
        for filename in filenames:
            filepaths.append(os.path.join(filepath, filename))
    return filepaths

if __name__ == '__main__':
    filepaths = get_filepaths(r"E:\文档\课程\大四\毕设\程序\data\backup-skyline")
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        df['anomaly_score'] = df['label']
        df['label'] = df['label'].apply(lambda x: 0)
        df.to_csv(filepath)