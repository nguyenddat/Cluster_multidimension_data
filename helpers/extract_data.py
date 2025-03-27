import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalized_data():
    df = pd.read_excel("./process_data/parent bedroom.xlsx")
    datatypes = [col for col in df.columns if col not in {"start_time", "end_time"}]

    normalized_df = df.copy()
    for datatype in datatypes:
        scaler = MinMaxScaler()
        
        rows = df[datatype].to_numpy().reshape(-1, 1)
        scaler.fit(rows)

        normalized_df[datatype] = scaler.transform(rows)

    ts_df = time_series_segment(normalized_df, datatypes)

    env_ts_df = ts_df.drop(columns = ["motion", "contact"])
    human_ts_df = ts_df[["date", "time", "motion", "contact"]]

    env_ts_df = df_to_numpy(env_ts_df)
    human_ts_df = df_to_numpy(human_ts_df)

    print(human_ts_df.isna().sum())
    env_ts_df = env_ts_df.drop(
        index=env_ts_df[
            env_ts_df["data"].apply(lambda x: x.shape != (6, 7))
        ].index
    ).reset_index(drop=True)

    human_ts_df = human_ts_df.drop(
        index=human_ts_df[
            human_ts_df["data"].apply(lambda x: x.shape != (2, 7))
        ].index
    ).reset_index(drop=True)

    env_data = np.stack(env_ts_df["data"].to_numpy())
    human_data = np.stack(human_ts_df["data"].to_numpy())

    return env_data, human_data


def derivative_data():
    df = pd.read_excel("./process_data/parent bedroom.xlsx")
    datatypes = [col for col in df.columns if col not in {"start_time", "end_time"}]

    raw_ts_df = time_series_segment(df, datatypes)
    
    env_raw_ts_df = raw_ts_df.drop(columns = ["motion", "contact"])
    human_raw_ts_df = raw_ts_df[["date", "time", "motion", "contact"]]

    env_raw_ts_df = df_to_numpy(env_raw_ts_df)
    human_raw_ts_df = df_to_numpy(human_raw_ts_df)

    env_raw_ts_df = env_raw_ts_df.drop(
        index=env_raw_ts_df[
            env_raw_ts_df["data"].apply(lambda x: x.shape != (6, 7))
        ].index
    ).reset_index(drop=True)

    human_raw_ts_df = human_raw_ts_df.drop(
        index=human_raw_ts_df[
            human_raw_ts_df["data"].apply(lambda x: x.shape != (2, 7))
        ].index
    ).reset_index(drop=True)

    env_raw_ts_data = np.stack(env_raw_ts_df["data"].tolist())
    human_raw_ts_data = np.stack(human_raw_ts_df["data"].tolist())

    env_derivative_data = np.gradient(env_raw_ts_data, axis=-1)
    human_derivative_data = np.gradient(human_raw_ts_data, axis=-1)

    return env_derivative_data, human_derivative_data, env_raw_ts_data, human_raw_ts_data
    

def df_to_numpy(df):
    datatypes = [col for col in df.columns if col not in {"date", "time"}]
    new_df = {"date": [], "time": [], "data": []}
    
    for idx, row in df.iterrows():
        new_df["date"].append(row["date"])
        new_df["time"].append(row["time"])
        
        data = [row[datatype] for datatype in datatypes]
        new_df["data"].append(np.vstack(data))

    return pd.DataFrame(new_df)


def time_series_segment(df, datatypes):
    morning_start = pd.to_datetime("05:00:00").time()
    morning_end = pd.to_datetime("12:00:00").time()

    afternoon_start = pd.to_datetime("13:00:00").time()
    afternoon_end = pd.to_datetime("20:00:00").time()

    dates = df["start_time"].dt.date.unique()
    df_data = {"date": [], "time": []}

    for datatype in datatypes:
        df_data[datatype] = []

    for date in dates:
        rows = df[
            df["start_time"].dt.date == date
        ]

        """---------------MORNING TIME---------------"""
        morning_df = rows[
            (rows["start_time"].dt.time >= morning_start) &
            (rows["start_time"].dt.time <= morning_end) &
            (rows["end_time"].dt.time <= morning_end)
        ].sort_values("start_time")

        for datatype in datatypes:
            df_data[datatype].append(
                np.array(
                    morning_df[datatype].tolist()
                )
            )

        df_data["date"].append(date)
        df_data["time"].append(f"{morning_start} - {morning_end}")

        """---------------AFTERNOON TIME--------------"""
        afternoon_df = rows[
            (rows["start_time"].dt.time >= afternoon_start) &
            (rows["start_time"].dt.time <= afternoon_end) &
            (rows["end_time"].dt.time <= afternoon_end)
        ]

        for datatype in datatypes:
            df_data[datatype].append(
                np.array(
                    afternoon_df[datatype].tolist()
                )
            )

        df_data["date"].append(date)
        df_data["time"].append(f"{afternoon_start} - {afternoon_end}")
    
    return pd.DataFrame(df_data)
