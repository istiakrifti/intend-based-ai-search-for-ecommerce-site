from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            
from constants import *

class Retriever:
    """
    A comprehensive retriever class that supports multiple retrieval strategies
    and maintains singleton pattern for system-wide usage.
    """
    _instance = None
    saved_retrievals = []
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Retriever, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, query_expansion_engine=None, config=None):
        """
        Initialize the retriever with various retrieval methods.
        
        Args:
            query_expansion_engine: The query expansion engine to use
            config: Configuration for the retriever
        """
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.query_expansion_engine = query_expansion_engine
            self.config = config or {}
            
            # Initialize different retrieval methods
            self.dense_retrievers = {}
            self.current_strategy = 'dense'
            self.n_retrievals = self.config.get('n_retrievals', 10)
            
            # Initialize vector store connections
            self._initialize_vector_stores()
    
    def _initialize_vector_stores(self):
        """Initialize connections to vector stores"""
        embedding_model_name = self.config.get('embedding_model', OPENAI_EMBEDDING_MODEL)
        vector_db_path = self.config.get('vector_db_path', 'vector_db')
        
        # Set up embeddings based on the model
        if 'bge' in embedding_model_name.lower():
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            embedding_model = HuggingFaceBgeEmbeddings(
                model_name=embedding_model_name, 
                model_kwargs=model_kwargs, 
                encode_kwargs=encode_kwargs
            )
        else:
            # Default to OpenAI embeddings if not BGE
            embedding_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        
        try:
            # Load FAISS vector database
            vector_db = FAISS.load_local(
                vector_db_path, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            
            # Create retriever from the vector database
            self.dense_retrievers['default'] = vector_db.as_retriever(
                search_kwargs={"k": self.n_retrievals}
            )
            
            print(f"Successfully loaded FAISS vector database from {vector_db_path}")
            
        except Exception as e:
            print(f"Failed to initialize FAISS vector store: {str(e)}")
    
    def set_query_expansion_engine(self, query_expansion_engine):
        """Set the query expansion engine"""
        self.query_expansion_engine = query_expansion_engine
        return self
    
    def set_strategy(self, strategy):
        """Set the current retrieval strategy"""
        valid_strategies = ['dense','fusion']
        if strategy in valid_strategies:
            self.current_strategy = strategy
            return True
        else:
            print(f"Invalid strategy: {strategy}. Valid options are: {valid_strategies}")
            return False
    
    def clear_saved_retrievals(self):
        """Clear saved retrievals"""
        self.saved_retrievals = []
    
    def retrieve(self, query_text, collection='default', n_retrievals=None):
        """
        Main retrieval method that uses the current strategy
        
        Args:
            query_text: The query text
            collection: The collection to search in
            n_retrievals: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if n_retrievals is None:
            n_retrievals = self.n_retrievals
        
        retrievals = []
        
        if self.current_strategy == 'dense':
            retrievals = self.dense_retrieve(query_text, collection, n_retrievals)
        elif self.current_strategy == 'fusion':
            retrievals = self.fusion_retrieve(query_text, collection, n_retrievals)
        else:
            print(f"Strategy {self.current_strategy} not implemented, using dense retrieval")
            retrievals = self.dense_retrieve(query_text, collection, n_retrievals)
        
        # Add query info to retrievals
        if isinstance(retrievals, list) and retrievals and not isinstance(retrievals[0], list):
            for i, r in enumerate(retrievals):
                r['rank'] = i + 1
                r['query'] = query_text
        
        # Save retrievals for logging/analysis
        self.saved_retrievals.append({
            'query': query_text,
            'strategy': self.current_strategy,
            'collection': collection,
            'retrievals': retrievals
        })
        
        return retrievals
    
    def dense_retrieve(self, query_text, collection='default', n_retrievals=None):
        """Vector search retrieval"""
        if n_retrievals is None:
            n_retrievals = self.n_retrievals
            
        if collection not in self.dense_retrievers:
            print(f"Collection {collection} not found in dense retrievers")
            return []
        
        retriever = self.dense_retrievers[collection]
        results = retriever.get_relevant_documents(query_text)[:n_retrievals]
        
        # Format results
        retrievals = []
        for i, doc in enumerate(results):
            # Extract score from metadata if available
            score = doc.metadata.get('score', 0) if hasattr(doc, 'metadata') else 0
            
            retrievals.append({
                'id': i + 1,
                'document': doc.page_content,
                'metadata': doc.metadata,
                'distance': score
            })
            
        return retrievals
    
    def fusion_retrieve(self, query_text, collection='default', n_retrievals=None):
        """RAG-Fusion with query expansion"""
        if n_retrievals is None:
            n_retrievals = self.n_retrievals
            
        if not self.query_expansion_engine:
            print("Query expansion engine not available, falling back to dense retrieval")
            return [self.dense_retrieve(query_text, collection, n_retrievals)]
        
        # Generate expanded queries
        try:
            expanded_queries = self.query_expansion_engine.expand_query(query_text)
        except Exception as e:
            print(f"Query expansion failed: {str(e)}")
            expanded_queries = [query_text]
        
        # Get results for each query
        all_results = []
        for expanded_query in expanded_queries:
            print(expanded_query)
            docs = self.dense_retrieve(expanded_query, collection, n_retrievals)
            all_results.append(docs)
        
        # Return raw results for reranking
        return all_results
    
    def show_retrievals(self,results, verbose=True):
        """
        Utility function to display results in a readable format,
        handling both retriever and reranker output formats.
        
        Args:
            results: List of results to display (can be retrievals or reranked results)
            verbose: Whether to print full document content
        """
        print('\n\nResults:\n')
        
        # Check if we're dealing with (doc, score) tuples from reranker
        if results and isinstance(results[0], tuple) and len(results[0]) == 2:
            # Handle reranked results format (list of (doc, score) tuples)
            for i, (doc, score) in enumerate(results):
                print(f'\n{i+1}.\nScore = {score:.4f}')
                
                # Extract document content based on its type
                if isinstance(doc, dict) and 'document' in doc:
                    document = doc['document'] if verbose else doc['document'][:100] + '...'
                    print(f'{document}\n')
                    if 'metadata' in doc and verbose:
                        print(f"Metadata: {doc['metadata']}")
                elif hasattr(doc, 'page_content'):
                    # It's a Document object
                    document = doc.page_content if verbose else doc.page_content[:100] + '...'
                    print(f'{document}\n')
                    if hasattr(doc, 'metadata') and verbose:
                        print(f"Metadata: {doc.metadata}")
                else:
                    # Other format
                    document = str(doc) if verbose else str(doc)[:100] + '...'
                    print(f'{document}\n')
        else:
            # Handle standard retriever format (list of dictionaries)
            for i, r in enumerate(results):
                if isinstance(r, dict) and 'document' in r:
                    document = r['document'] if verbose else r['document'][:100] + '...'
                    distance = r.get('distance', r.get('score', 'N/A'))
                    
                    print(f'\n{i+1}.\nDistance/Score = {distance}')
                    print(f'{document}\n')
                    if 'metadata' in r and verbose:
                        print(f"Metadata: {r['metadata']}")
                else:
                    # Handle unexpected format
                    print(f'\n{i+1}.\n{str(r)[:200]}...')
# Example usage
if __name__ == "__main__":
    from reranker import ReRanker
    from queryexpansion import QueryExpansionEngine
    
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
    retriever = Retriever(
        query_expansion_engine=query_expansion,
        config=config
    )
    retriever.set_strategy('fusion')  # Or 'dense' 
    # Step 1: Retrieve documents using the Retriever
    query = "gaming laptop with good graphics"
    retrieved_docs = retriever.retrieve(query)
    
    print("=== Retrieved Documents (Before Reranking) ===")
    if retriever.current_strategy == 'dense':
        retriever.show_retrievals(retrieved_docs)
    else:
        print("Showing first set of fusion results (from expanded query):")
        retriever.show_retrievals(retrieved_docs[0])
    
    # Option 2: General rerank method with any strategy
    # Initialize reranker
    reranker = ReRanker(strategy="bge")
    reranked_docs = reranker.rerank(
        retrieved_docs, 
        query, 
        top_k=5, 
    )

    print("=== Retrieved Documents (After Reranking) ===")

    reranker.get_documents_only(reranked_docs)
    
    