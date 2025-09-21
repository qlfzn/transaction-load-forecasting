"""
This module covers data extraction part of the project, mainly includes how data is pulled from source data (csv) and being fed into the pipeline
"""
from src.services import SparkService

class Extractor:
    def __init__(self) -> None:
        self.spark = SparkService().init_spark()

    def read_batch(self, source_path: str):
        return self.spark.read.csv(path=source_path, header=True, inferSchema=True)