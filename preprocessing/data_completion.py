import pandas as pd
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from preprocessing.conversions import Columns


"""
this file contains different data completion strategies
"""


class DataComplete(ABC):
    @abstractmethod
    def complete(self, data: pd.DataFrame):
        ...


class Clustering(DataComplete):
    EXCLUDE_COLUMNS = [
        Columns.DIAGNOSIS_DATE
    ]

    def __init__(self, num_columns: int = 8, k: int = 10):
        self.num_columns = num_columns
        self.k = k

    def complete(self, data: pd.DataFrame):
        columns = list(data.columns)
        columns = [col for col in columns if col not in self.EXCLUDE_COLUMNS]
        columns.sort(key=lambda col: data[col].isna().sum())
        columns = columns[:self.num_columns]
        clustering_features = data[columns].copy()
        clustering_features = clustering_features.fillna(
            clustering_features.median(numeric_only=True))

        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(clustering_features)

        kmeans = KMeans(n_clusters=self.k, random_state=42)
        clusters = kmeans.fit_predict(X_cluster)

        # Add cluster labels to original DataFrame
        data["cluster"] = clusters
        # data.to_csv("temp.csv", index=False, encoding='utf-8-sig')
        for col in data.columns:
            for i in range(self.k):
                mean_val = data.loc[(data['cluster'] == i) & (
                    data[col].notna()), col].mean()
                if pd.isna(mean_val):
                    mean_val = data[col].median()
                    if pd.isna(mean_val):
                        mean_val = 0
                data.loc[(data['cluster'] == i) & (data[col].isna()), col] = mean_val

        data.drop(columns="cluster", inplace=True)

        return data


class Mean(DataComplete):
    def complete(self, data: pd.DataFrame):
        for col in data.columns:
            mean_val = data[col].mean()
            if pd.isna(mean_val):
                mean_val = 0
            data[col] = data[col].fillna(mean_val)
        # data = data.apply(lambda col: col.fillna(
        #     col.mean()) if pd.api.types.is_numeric_dtype(col) else col)
        return data


class DefaultValue(DataComplete):
    def __init__(self, default_value: int = 0):
        self.default_value = default_value

    def complete(self, data: pd.DataFrame):
        data.fillna(self.default_value, inplace=True)
        return data
