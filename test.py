from reranker import ReRanker
from retriever import Retriever
from queryexpansion import QueryExpansionEngine
from constants import *

def retrieve_and_rerank(query, retriever, reranker, top_k=5):
    """
    Utility function that connects retriever and reranker without coupling them.
    
    Args:
        query: The query text
        retriever: An initialized Retriever instance
        reranker: An initialized ReRanker instance
        top_k: Number of top documents to return after reranking
        
    Returns:
        List of reranked documents with their scores
    """
    # Step 1: Get raw results from the retriever
    retrieved_docs = retriever.retrieve(query)
    print("Before")
    retriever.show_retrievals(retrieved_docs)
    print("After")    
    # Step 2: Process based on retrieval strategy
    if retriever.current_strategy == 'fusion':
        # For fusion strategy, retrieved_docs is already a list of lists
        reranked_results = reranker.rerank(retrieved_docs, query=query)
    else:
        # For dense strategy, we need to wrap the results in a list
        reranked_results = reranker.rerank([retrieved_docs], query=query)
    
    # Step 3: Return top_k results or all if top_k is None
    if top_k is not None:
        return reranked_results[:top_k]
    return reranked_results

# Example usage:
if __name__ == "__main__":
    # Configuration
    config = {
        'vector_db_path': DB_PATH,
        'embedding_model': OPENAI_EMBEDDING_MODEL,
        'n_retrievals': 10
    }
    
    # Initialize query expansion engine
    query_expansion = QueryExpansionEngine(
        num_queries=4,
        model_name=QUERY_EXPANSION_MODEL
    )
    
    # Initialize retriever
    retriever = Retriever(
        query_expansion_engine=query_expansion,
        config=config
    )
    
    # Set retrieval strategy
    retriever.set_strategy('fusion')  # Or 'dense'
    
    # Initialize reranker (completely separate from retriever)
    reranker = ReRanker(strategy="hybrid_rrf_bge")
    
    # Perform retrieval and reranking
    query = "gaming laptop with good graphics"
    
    # Get reranked results
    reranked_results = retrieve_and_rerank(query, retriever, reranker, top_k=5)
    
    # Display results
    print("=== Reranked Results ===")
    for idx, (doc, score) in enumerate(reranked_results):
        print(f"\n{idx+1}. Score: {score:.4f}")
        if isinstance(doc, dict) and 'document' in doc:
            print(f"Document: {doc['document'][:150]}...")
        else:
            print(f"Document: {str(doc)[:150]}...")
    
    # If you only want the documents without scores
    docs_only = reranker.get_documents_only(reranked_results)
    
    # Format them for use with show_retrievals if needed
    formatted_docs = []
    for doc in docs_only:
        if isinstance(doc, dict) and 'document' in doc:
            formatted_docs.append(doc)
        else:
            # Convert to the format expected by show_retrievals
            formatted_doc = {
                'id': getattr(doc, 'id', 'unknown') if hasattr(doc, 'metadata') else 'unknown',
                'document': doc.page_content if hasattr(doc, 'page_content') else str(doc),
                'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
            }
            formatted_docs.append(formatted_doc)
    
    # Show using retriever's method
    retriever.show_retrievals(formatted_docs)