
from .in_memory_repo import InMemoryRepository
from .datasets import AdultIncomeDataset, MedicalCostDataset
from .loaders import PandasCsvLoader
from .network_factory import ModelFactory


repo = InMemoryRepository()
repo.add_dataset("adult_income", AdultIncomeDataset(csv_loader=PandasCsvLoader()))
repo.add_dataset("medical_cost", MedicalCostDataset(csv_loader=PandasCsvLoader()))

def get_repo() -> InMemoryRepository:
    return repo

model_factory = ModelFactory()

def get_model_factory() -> ModelFactory:
    return model_factory
