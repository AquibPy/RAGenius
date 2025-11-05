from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, \
    TextLoader, CSVLoader, \
    Docx2txtLoader, JSONLoader, UnstructuredExcelLoader, PyPDFLoader

def load_all_documents(data_directory: str):
    LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
    ".json": JSONLoader,
    ".xlsx": UnstructuredExcelLoader,
    # Add more formats here as needed
    }
    
    # Define a dynamic loader selector
    def dynamic_loader(file_path: str):
        ext = Path(file_path).suffix.lower()
        loader_cl = LOADER_MAP.get(ext)
        if not loader_cl:
            raise ValueError(f"❌ Unsupported file type: {file_path}")
        return loader_cl(file_path)

    # Create a single DirectoryLoader that handles all formats
    loader = DirectoryLoader(
        data_directory,
        glob="**/*.*",          # Include all file types
        loader_cls=dynamic_loader,
        show_progress=True
    )

    print("📂 Scanning directory for supported files...")
    all_documents = loader.load()
    print(f"✅ Total documents loaded: {len(all_documents)}")

    # Add metadata (file name + type)
    for doc in all_documents:
        path = Path(doc.metadata["source"])
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = path.suffix.lower().replace(".", "")
    return all_documents

if  __name__=='__main__':
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs if docs else None)