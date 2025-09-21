from src.pipeline import MLPipeline

if __name__ == "__main__":
    pipeline = MLPipeline()
    df = pipeline.run_extract_data()
    df.show(5)
    print(f"Total rows: {df.count()}")