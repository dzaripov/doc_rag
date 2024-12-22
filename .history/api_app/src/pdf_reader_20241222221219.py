from langchain_community.document_loaders import PyPDFLoader

def read_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()