"""
Run ML pipeline
"""
from src.extract import Extractor

class MLPipeline:
    def __init__(self) -> None:
        self.extractor = Extractor()

    def run_extract_data(self):
       df = self.extractor.read_batch("./data/synthetic_fraud_data.csv")
       return df