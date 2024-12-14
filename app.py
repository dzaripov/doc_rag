from src.pipeline import RAGPipeline

texts = None # here load and preprocess texts

rag_pipeline = RAGPipeline(config_path='config', texts=texts)