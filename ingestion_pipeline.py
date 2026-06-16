import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} not found")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError("No .txt files found")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Length: {len(doc.page_content)}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store created at {persist_directory}")
    return vectorstore


def main():
    print("🚀 Starting ingestion pipeline...")

    docs_path = "docs"
    persist_directory = "db/chroma_db"

    # Step 1: Load documents
    documents = load_documents(docs_path)

    # Step 2: Split documents
    chunks = split_documents(documents)

    # Step 3: Create vector DB
    vectorstore = create_vector_store(chunks, persist_directory)

    print("\n✅ Ingestion complete! Ready for RAG queries.")

    return vectorstore


if __name__ == "__main__":
    main()