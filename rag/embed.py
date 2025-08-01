# Loads PDFs, creates vector store, saves it to embeddings/


# in case of need to re-embedd
# python rag/embed.py --force

# rag/embed.py
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# first load in the environments to the page
# need the api key for embedding given by openAPI key..
load_dotenv()

# so need to place where getting them and need to put 
# where to store once we embedd using (Facebook Index Similarity Search — FAISS)
DATA_DIR = "data"
EMBEDDING_DIR = "embeddings/index"

# built a function 
# checking if this extension exists, if not and force is not passed, then skip
def build_index(force=False):
    if os.path.exists(EMBEDDING_DIR) and not force:
        print("Embeddings already exist. Skipping re-embedding.")
        return

    # Get all PDF files in /data
    pdf_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

    # the PyPDF loader will get the extension (PDF file), need to load this in 
    # and back in the FAISS docs the chunks and embeddings 
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        source_name = os.path.basename(path).replace(".pdf", "").replace("_", " ").title()

        # Tag each page with a human-readable source name
        for doc in pages:
            doc.metadata["source"] = source_name
        all_docs.extend(pages)

    # Split the text into manageable chunks for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    print(f"Total chunks created: {len(chunks)}")

    # Generate vector embeddings for each chunk using OpenAI
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)
    
    # Save the vector database locally (to embeddings/index/)
    vectordb.save_local(EMBEDDING_DIR)
    print(f"Embeddings saved to {EMBEDDING_DIR}")

# If this script is run directly, execute the index build
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-embedding even if index exists")
    args = parser.parse_args()
    build_index(force=args.force)

