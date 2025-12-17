import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df['user'] = user_encoder.fit_transform(df['user_id'])
    df['item'] = item_encoder.fit_transform(df['asin'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['user', 'timestamp'])

    user_histories = defaultdict(list)
    for row in df.itertuples():
        user_histories[row.user].append(row.item)

    return df, user_histories