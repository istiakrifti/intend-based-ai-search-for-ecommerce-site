from reranker import ReRanker
from retriever import Retriever
from queryexpansion import QueryExpansionEngine
from constants import *

def retrieve_and_rerank(query, retriever, reranker, sql_documents, top_k=5):
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
        all_metadata: List to collect metadata from the results
        verbose: Whether to collect full document content (default is True)
    """
    all_metadata = []
    # Check if we're dealing with (doc, score) tuples from reranker
    if results and isinstance(results[0], tuple) and len(results[0]) == 2:
        # Handle reranked results format (list of (doc, score) tuples)
        for i, (doc, score) in enumerate(results):
            # Extract document content based on its type
            if isinstance(doc, dict) and 'document' in doc:
                document = doc['document'] if verbose else doc['document'][:100] + '...'
                all_metadata.append({'id': f"sql_{i + 1}", 'document': document, 'metadata': doc.get('metadata', {}), 'distance': score})
            elif hasattr(doc, 'page_content'):
                # It's a Document object
                document = doc.page_content if verbose else doc.page_content[:100] + '...'
                all_metadata.append({'id': f"sql_{i + 1}", 'document': document, 'metadata': getattr(doc, 'metadata', {}), 'distance': score})
            else:
                # Other format
                document = str(doc) if verbose else str(doc)[:100] + '...'
                all_metadata.append({'id': f"sql_{i + 1}", 'document': document, 'metadata': 'N/A', 'distance': score})
    else:
        # Handle standard retriever format (list of dictionaries)
        for i, r in enumerate(results):
            if isinstance(r, dict) and 'document' in r:
                document = r['document'] if verbose else r['document'][:100] + '...'
                distance = r.get('distance', r.get('score', 'N/A'))
                all_metadata.append({'id': f"sql_{i + 1}", 'document': document, 'metadata': r.get('metadata', {}), 'distance': distance})
            else:
                # Handle unexpected format
                all_metadata.append({'id': f"sql_{i + 1}", 'document': str(r)[:100] + '...', 'metadata': 'N/A', 'distance': 'N/A'})

    return all_metadata

def final_retrieval(query, sql_documents):
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
    
    # Set retrieval strategy
    retriever.set_strategy('dense')  # Or 'dense'
    
    # Initialize reranker (completely separate from retriever)
    reranker = ReRanker(strategy="cross_encoder")
    
    # Get reranked results
    reranked_results = retrieve_and_rerank(query, retriever, reranker, sql_documents, top_k=5)
    
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
    all_metadata = get_result(formatted_docs)

    return {"final_result": all_metadata}