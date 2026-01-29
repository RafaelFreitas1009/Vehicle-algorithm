# data/processor.py
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    """Processamento e preparação dos dados (limpeza, encoding, transformações)."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df_original = pd.read_csv(filepath)
        self.df: pd.DataFrame | None = None
        self.df_num: pd.DataFrame | None = None
        self.encoders: dict[str, LabelEncoder] = {}
        self.age_map = {
            "below21": 20, "21": 21, "26": 26, "31": 31,
            "36": 36, "41": 41, "46": 46, "50plus": 55
        }

    def process_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Processa os dados: limpeza, tratamento de nulos e encoding."""
        self.df = self.df_original.drop(columns=["car"]).copy()

        # preencher nulos com moda
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        # cópia numérica
        self.df_num = self.df.copy()

        # mapear idade
        if "age" in self.df_num.columns:
            self.df_num["age"] = self.df_num["age"].map(self.age_map)

        # label encoding para variáveis categóricas
        for col in self.df_num.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            self.df_num[col] = le.fit_transform(self.df_num[col])
            self.encoders[col] = le

        return self.df, self.df_num

    def get_unique_values(self, column: str):
        if self.df is None:
            raise RuntimeError("Você precisa rodar process_data() antes.")
        return self.df[column].unique()
