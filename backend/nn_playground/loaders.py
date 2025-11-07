
from abc import ABC, abstractmethod
import pandas as pd


class BaseCsvLoader(ABC):
    @abstractmethod
    def load(self, path: str, **kwargs) -> pd.DataFrame: ...


class PandasCsvLoader(BaseCsvLoader):
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)

