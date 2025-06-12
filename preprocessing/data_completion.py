import pandas as pd
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class DataComplete(ABC):
    @abstractmethod
    def complete(self, data: pd.DataFrame):
        ...


class Clustering(DataComplete):
    def __init__(self, num_columns: int = 10, k: int = 50):
        self.num_columns = num_columns
        self.k = k

    def complete(self, data: pd.DataFrame):
        columns = list(data.columns)
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

        for col in data.columns:
            if data[col].isna().sum() > 0:
                data[col] = data.groupby("cluster")[col].transform(
                    lambda x: x.fillna(x.median())  # or mean(), mode(), etc.
                )

        data.drop("cluster", inplace=True)

        return data


class Mean(DataComplete):
    def complete(self, data: pd.DataFrame):
        data = data.apply(lambda col: col.fillna(
            col.mean()) if pd.api.types.is_numeric_dtype(col) else col)
        return data


class DefaultValue(DataComplete):
    def __init__(self, default_value: int = 0):
        self.default_value = default_value

    def complete(self, data: pd.DataFrame):
        data.fillna(self.default_value, inplace=True)
        return data
