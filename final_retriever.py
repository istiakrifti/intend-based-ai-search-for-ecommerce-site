from reranker import ReRanker
from retriever import Retriever
from queryexpansion import QueryExpansionEngine
from constants import *

def retrieve_and_rerank(query, retriever, reranker, sql_documents, top_k=5, retrieved_docs=None):
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
    # retrieved_docs = retriever.retrieve(query)
    
    print("Before")
    # retriever.show_retrievals(retrieved_docs)
    print("After")    
    # Step 2: Process based on retrieval strategy
    if retriever.current_strategy == 'fusion':
        # For fusion strategy, retrieved_docs is already a list of lists
        if sql_documents is not None:
            for sublist in retrieved_docs:
                sublist.extend(sql_documents)
        reranked_results = reranker.rerank(retrieved_docs, query=query)
    else:
        # For dense strategy, we need to wrap the results in a list
        if sql_documents is not None:
            retrieved_docs.extend(sql_documents)
        reranked_results = reranker.rerank([retrieved_docs], query=query)
    
    # Step 3: Return top_k results or all if top_k is None
    if top_k is not None:
        return reranked_results[:top_k]
    return reranked_results

def get_result(results, verbose=True):
    """
    Utility function to collect results (documents and metadata) in a list format,
    handling both retriever and reranker output formats without printing.
    
    Args:
        results: List of results to process (can be retrievals or reranked results)
        verbose: Whether to collect full document content (default is True)
    """
    all_metadata = []

    if results and isinstance(results[0], tuple) and len(results[0]) == 2:
        # Reranked format: list of (doc, score) tuples
        for i, (doc, score) in enumerate(results):
            if isinstance(doc, dict) and 'document' in doc:
                all_metadata.append(doc.get('metadata', {}))
            elif hasattr(doc, 'page_content'):
                all_metadata.append(getattr(doc, 'metadata', {}))
            else:
                all_metadata.append({})
    else:
        # Standard retriever format
        for i, r in enumerate(results):
            if isinstance(r, dict) and 'document' in r:
                all_metadata.append(r.get('metadata', {}))
            elif hasattr(r, 'page_content'):
                all_metadata.append(getattr(r, 'metadata', {}))
            else:
                all_metadata.append({})

    return all_metadata


# Configuration
config = {
    'vector_db_path': DB_PATH,
    'embedding_model': OPENAI_EMBEDDING_MODEL,
    'n_retrievals': 6
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


def final_retrieval(query):
    # # Configuration
    # config = {
    #     'vector_db_path': DB_PATH,
    #     'embedding_model': OPENAI_EMBEDDING_MODEL,
    #     'n_retrievals': 6
    # }
    
    # # Initialize query expansion engine
    # query_expansion = QueryExpansionEngine(
    #     num_queries=4,
    #     model_name=QUERY_EXPANSION_MODEL
    # )
    
    # # Initialize retriever
    # retriever = Retriever(
    #     query_expansion_engine=query_expansion,
    #     config=config
    # )
    
    # Set retrieval strategy
    retriever.set_strategy('dense')  # Or 'dense'
    
    # Initialize reranker (completely separate from retriever)
    # reranker = ReRanker(strategy="cross_encoder")
    retrieved_docs = retriever.retrieve(query)

    return retrieved_docs

def rerank(query, sql_documents, retrieved_docs):
    reranker = ReRanker(strategy="cross_encoder")
    # Get reranked results
    reranked_results = retrieve_and_rerank(query, retriever, reranker, sql_documents, top_k=5, retrieved_docs=retrieved_docs)
    
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
    
    # retriever.show_retrievals(formatted_docs)
    all_metadata = get_result(formatted_docs)

    return all_metadata