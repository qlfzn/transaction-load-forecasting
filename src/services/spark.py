"""
Initiates spark instance 
"""
from pyspark.sql import SparkSession

class SparkService:
    def __init__(self) -> None:
        """
        Initiate spark session
        """
        self.spark = SparkSession \
                        .builder \
                        .appName("ML Pipeline") \
                        .config("spark.executor.memory", "2g") \
                        .config("spark.executor.cores", "2") \
                        .config("spark.dynamicAllocation.enabled", "true") \
                        .getOrCreate()

    def init_spark(self):
        return self.spark