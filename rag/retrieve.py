# Loads vector store, runs similarity search on user question
# Loads the saved FAISS index from embeddings/index/
# Retrieves relevant chunks for a given query
# Returns both:the retrieved text and a list of source names for citation (e.g. "ADA Guidelines", "CDC Oral Health")

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# function to load pre-embedded chunks and get top-k relevant ones
def retrieve_context(query, k=3):
    
    
    
    # load from local FAISS index
    db = FAISS.load_local("embeddings/index", OpenAIEmbeddings())

    # search top-k matching chunks - set here default to top (3) chunks
    results = db.similarity_search(query, k=k)

    # collect plain text context
    context_chunks = []
    # ensuring doesnt give the same sources using a set
    sources = set()

    for doc in results:
        context_chunks.append(doc.page_content)
        if "source" in doc.metadata:
            sources.add(doc.metadata["source"])

    full_context = "\n\n".join(context_chunks)
    return full_context, sorted(list(sources))
