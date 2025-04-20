from typing import List, Dict, Any, Tuple, Callable, Optional, Union
from langchain.load import dumps, loads
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker
import numpy as np
from constants import *

class ReRanker:
    def __init__(self, 
                 strategy: str = "reciprocal_rank_fusion", 
                 cross_encoder_model: str = CROSS_RERANKER_MODEL,
                 bge_model: str = BGE_RERANKER_MODEL,
                 use_fp16: bool = True,
                 batch_size: int = 8):
        """
        Initialize the ReRanker with a specific strategy.
        
        Args:
            strategy: Reranking strategy. Options: 
                    "reciprocal_rank_fusion", "cross_encoder", "bge", "hybrid_rrf_ce", "hybrid_rrf_bge"
            cross_encoder_model: The cross-encoder model to use for semantic reranking
            bge_model: The BGE reranker model to use
            use_fp16: Whether to use half-precision for BGE model
            batch_size: Batch size for processing multiple items
        """
        self.strategy = strategy
        self.batch_size = batch_size
        
        # Initialize models based on selected strategy
        if strategy in ["cross_encoder", "hybrid_rrf_ce"]:
            self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        if strategy in ["bge", "hybrid_rrf_bge"]:
            self.bge_reranker = FlagReranker(bge_model, use_fp16=use_fp16)
        
        # Map strategies to their implementation functions
        self.strategy_map = {
            "reciprocal_rank_fusion": self.reciprocal_rank_fusion,
            "cross_encoder": self.cross_encoder_rerank,
            "bge": self.bge_rerank,
            "hybrid_rrf_ce": self.hybrid_rrf_ce,
            "hybrid_rrf_bge": self.hybrid_rrf_bge
        }
    
    def reciprocal_rank_fusion(self, 
                              results: List[List[Union[Document, Dict[str, Any]]]], 
                              k: int = 60,
                              **kwargs) -> List[Tuple[Union[Document, Dict[str, Any]], float]]:
        """
        Args:
            results: List of ranked document lists
            k: Constant in RRF formula to stabilize scores 
        Returns:
            Reranked documents with their fusion scores
        """
        fused_scores = {}

        # Process each result list
        for docs in results:
            # Process each document with its rank
            for rank, doc in enumerate(docs):
                # Convert document to string for dict key
                doc_str = dumps(doc)
                print("Rank", rank)
                # Initialize score if first occurrence
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                
                # Add RRF score: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)
        
        # Sort by scores and convert back from strings
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return reranked_results
    
    def _extract_text(self, doc: Union[Document, Dict[str, Any]]) -> str:
        """Extract text content from different document formats."""
        if isinstance(doc, Document):
            return doc.page_content
        elif isinstance(doc, dict) and "page_content" in doc:
            return doc["page_content"]
        elif isinstance(doc, dict) and "text" in doc:
            return doc["text"]
        else:
            # Try to convert the entire document to a string as fallback
            return str(doc)
    
    def _prepare_for_reranking(self, results: List[List[Union[Document, Dict[str, Any]]]]) -> List[Union[Document, Dict[str, Any]]]:
        """Prepare and deduplicate documents for reranking."""
        # Flatten the list of lists into a single list of documents
        flattened_docs = []
        for docs in results:
            flattened_docs.extend(docs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in flattened_docs:
            doc_str = dumps(doc)
            if doc_str not in seen:
                seen.add(doc_str)
                unique_docs.append(doc)
                
        return unique_docs
    
    def cross_encoder_rerank(self, 
                            results: List[List[Union[Document, Dict[str, Any]]]],
                            query: str,
                            **kwargs) -> List[Tuple[Union[Document, Dict[str, Any]], float]]:
        """
        Rerank documents using a cross-encoder model.
        
        Args:
            results: List of ranked document lists (will be flattened)
            query: The original query for reranking
            
        Returns:
            Reranked documents with relevance scores
        """
        # Get unique documents
        unique_docs = self._prepare_for_reranking(results)
        
        # Prepare pairs for the cross-encoder
        pairs = []
        for doc in unique_docs:
            doc_text = self._extract_text(doc)
            pairs.append([query, doc_text])
        
        # Process in batches to prevent OOM errors
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores = self.cross_encoder.predict(batch)
            all_scores.extend(scores)
        
        # Create reranked results
        reranked_results = [(doc, float(score)) for doc, score in zip(unique_docs, all_scores)]
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results
    
    def bge_rerank(self, 
                  results: List[List[Union[Document, Dict[str, Any]]]],
                  query: str,
                  **kwargs) -> List[Tuple[Union[Document, Dict[str, Any]], float]]:
        """
        Rerank documents using the BGE reranker.
        
        Args:
            results: List of ranked document lists (will be flattened)
            query: The original query for reranking
            
        Returns:
            Reranked documents with relevance scores
        """
        if self.bge_reranker is None:
            # Fall back to cross-encoder if BGE isn't available
            return self.cross_encoder_rerank(results, query, **kwargs)
        
        # Get unique documents
        unique_docs = self._prepare_for_reranking(results)
        
        # Prepare pairs for the BGE reranker
        pairs = []
        for doc in unique_docs:
            doc_text = self._extract_text(doc)
            pairs.append([query, doc_text])
        
        # Process in batches
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            try:
                scores = self.bge_reranker.compute_score(batch)
                all_scores.extend(scores)
            except Exception as e:
                print(f"Error during BGE reranking: {str(e)}")
                # Assign neutral scores if reranking fails
                all_scores.extend([0.5] * len(batch))
        
        # Create reranked results
        reranked_results = [(doc, float(score)) for doc, score in zip(unique_docs, all_scores)]
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results
    
    def hybrid_rrf_ce(self,
                     results: List[List[Union[Document, Dict[str, Any]]]],
                     query: str,
                     rrf_k: int = 60,
                     alpha: float = 0.5,
                     **kwargs) -> List[Tuple[Union[Document, Dict[str, Any]], float]]:
        """
        Hybrid reranking combining RRF with cross-encoder scores.
        
        Args:
            results: List of ranked document lists
            query: The original query for reranking
            rrf_k: Constant for RRF calculation
            alpha: Weight for cross-encoder score (1-alpha for RRF score)
            
        Returns:
            Reranked documents with hybrid scores
        """
        # Get RRF scores
        rrf_results = self.reciprocal_rank_fusion(results, k=rrf_k)
        rrf_docs = [doc for doc, _ in rrf_results]
        rrf_scores_map = {dumps(doc): score for doc, score in rrf_results}
        
        # Get cross-encoder scores
        cross_results = self.cross_encoder_rerank([rrf_docs], query=query)
        
        # Combine scores
        hybrid_results = []
        for doc, ce_score in cross_results:
            doc_str = dumps(doc)
            rrf_score = rrf_scores_map[doc_str]
            
            # Normalize RRF score to [0,1] range (approximate)
            max_rrf = 1.0  # Maximum possible RRF score is roughly 1.0
            norm_rrf_score = min(rrf_score / max_rrf, 1.0)
            
            # Calculate hybrid score
            hybrid_score = (alpha * ce_score) + ((1 - alpha) * norm_rrf_score)
            hybrid_results.append((doc, hybrid_score))
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_results
    
    def hybrid_rrf_bge(self,
                    results: List[List[Union[Document, Dict[str, Any]]]],
                    query: str,
                    rrf_k: int = 60,
                    alpha: float = 0.5,
                      **kwargs) -> List[Tuple[Union[Document, Dict[str, Any]], float]]:
        """
        Hybrid reranking combining RRF with BGE reranker scores.
        
        Args:
            results: List of ranked document lists
            query: The original query for reranking
            rrf_k: Constant for RRF calculation
            alpha: Weight for BGE score (1-alpha for RRF score)
            
        Returns:
            Reranked documents with hybrid scores
        """
        if self.bge_reranker is None:
            # Fall back to hybrid RRF + cross-encoder if BGE isn't available
            return self.hybrid_rrf_ce(results, query, rrf_k, alpha, **kwargs)
        
        # Get RRF scores
        rrf_results = self.reciprocal_rank_fusion(results, k=rrf_k)
        rrf_docs = [doc for doc, _ in rrf_results]
        rrf_scores_map = {dumps(doc): score for doc, score in rrf_results}
        
        # Get BGE scores
        bge_results = self.bge_rerank([rrf_docs], query=query)
        
        # Combine scores
        hybrid_results = []
        for doc, bge_score in bge_results:
            doc_str = dumps(doc)
            rrf_score = rrf_scores_map[doc_str]
            
            # Normalize RRF score to [0,1] range (approximate)
            max_rrf = 1.0  # Maximum possible RRF score is roughly 1.0
            norm_rrf_score = min(rrf_score / max_rrf, 1.0)
            
            # Calculate hybrid score
            hybrid_score = (alpha * bge_score) + ((1 - alpha) * norm_rrf_score)
            hybrid_results.append((doc, hybrid_score))
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_results
    
    def rerank(self, 
            results: List[List[Union[Document, Dict[str, Any]]]],
            query: Optional[str] = None,
              **kwargs) -> List[Tuple[Union[Document, Dict[str, Any]], float]]:
        """
        Rerank documents using the selected strategy.
        
        Args:
            results: List of ranked document lists
            query: The original query (required for all strategies except RRF)
            **kwargs: Additional parameters for specific reranking strategies
            
        Returns:
            Reranked documents with scores
        """
        if self.strategy not in self.strategy_map:
            raise ValueError(f"Strategy '{self.strategy}' not supported")
        
        if self.strategy != "reciprocal_rank_fusion" and query is None:
            raise ValueError(f"Query parameter is required for '{self.strategy}' strategy")
            
        return self.strategy_map[self.strategy](results, query=query, **kwargs)
    
    def get_documents_only(self, 
                        reranked_results: List[Tuple[Union[Document, Dict[str, Any]], float]], 
                        top_k: Optional[int] = None) -> List[Union[Document, Dict[str, Any]]]:
        """
        Extract just the documents from reranked results.
        
        Args:
            reranked_results: Reranked results with scores
            top_k: Number of top results to return
            
        Returns:
            List of documents without scores
        """
        docs = [doc for doc, _ in reranked_results]
        if top_k is not None:
            docs = docs[:top_k]
        return docs
    

# Example usage
if __name__ == "__main__":
    from typing import List, Dict, Any, Tuple, Callable, Optional, Union
    
    
    
    # Sample candidates (would come from your initial retrieval)
    candidates = [
        {"id": "1005", "name": "Acer Nitro 5 AN515-45-R7BF Ryzen 5 16GB RAM", 
         "category": "Laptop", "subcategory": "Gaming_Laptop", "brand": "Acer", "similarity_score": 0.78},
        {"id": "1006", "name": "Dell XPS 13 Intel Core i7 16GB RAM", 
         "category": "Laptop", "subcategory": "Ultrabook", "brand": "Dell", "similarity_score": 0.75},
        {"id": "1007", "name": "Apple MacBook Pro M1 8GB RAM", 
         "category": "Laptop", "subcategory": "Professional", "brand": "Apple", "similarity_score": 0.72},
        {"id": "1008", "name": "MSI GF63 Gaming Laptop NVIDIA RTX 3050", 
         "category": "Laptop", "subcategory": "Gaming_Laptop", "brand": "MSI", "similarity_score": 0.68},
        {"id": "1009", "name": "Lenovo Legion 5 AMD Ryzen 7 RTX 3060", 
         "category": "Laptop", "subcategory": "Gaming_Laptop", "brand": "Lenovo", "similarity_score": 0.65}
    ]
    
    # Original query
    query = "gaming laptop with good graphics"
    
    # Simulate multiple query results as would happen in RAG-Fusion
    # Generate a few variations of the original search to demonstrate RRF
    query_variations = [
        "gaming laptop with powerful graphics card",
        "best laptops for playing games",
        "high performance gaming notebooks",
        "laptops with good GPU for gaming"
    ]
    
    # Generate multiple result sets (simulating retrieval for each query)
    results = []
    for i, _ in enumerate(query_variations):
        # Simulate different ordering based on different queries
        # by shuffling the candidates slightly for each variation
        import random
        shuffled = candidates.copy()
        random.shuffle(shuffled)
        results.append(shuffled)
    
    print(f"Original query: '{query}'")
    print("-----------------------------------")
    
    # Initialize the reranker with cross-encoder strategy (most reliable)
    reranker = ReRanker(
        strategy="bge",
        cross_encoder_model = CROSS_RERANKER_MODEL,
        batch_size=16
    )
    print(f"Original query: '{query}'")
    print("-----------------------------------")
    
    # print("\n1. Cross-encoder reranking:")
    # reranked = reranker.cross_encoder_rerank(results, query=query)
    # for doc, score in reranked[:3]:
    #     print(f"Score: {score:.4f} - {doc['name']} ({doc['brand']}, {doc['subcategory']})")
    
    # print("\n2. Reciprocal Rank Fusion:")
    # rrf_results = reranker.reciprocal_rank_fusion(results)
    # for doc, score in rrf_results[:3]:
    #     print(f"Score: {score:.4f} - {doc['name']} ({doc['brand']}, {doc['subcategory']})")
    
    print("\n3. Hybrid (RRF + bge):")
    hybrid_results = reranker.hybrid_rrf_bge(results, query=query)
    for doc, score in hybrid_results[:3]:
        print(f"Score: {score:.4f} - {doc['name']} ({doc['brand']}, {doc['subcategory']})")
    
    # print("\n4. Trying BGE reranker (will fallback to cross-encoder if BGE not available):")
    # try:
    #     # Create a new reranker with BGE strategy
    #     bge_reranker = ReRanker(strategy="bge", bge_model="BAAI/bge-reranker-base")
    #     bge_results = bge_reranker.rerank(results, query=query)
    #     for doc, score in bge_results[:3]:
    #         print(f"Score: {score:.4f} - {doc['name']} ({doc['brand']}, {doc['subcategory']})")
    # except Exception as e:
    #     print(f"BGE reranking failed, using cross-encoder instead: {str(e)}")
    #     bge_results = reranker.rerank(results, query=query)
    #     for doc, score in bge_results[:3]:
    #         print(f"Score: {score:.4f} - {doc['name']} ({doc['brand']}, {doc['subcategory']})")

    # reranker = ReRanker()
    # recp_results = reranker.rerank(results, query=query)
    # for doc, score in recp_results[:3]:
    #     print(f"Score: {score:.4f} - {doc['name']} ({doc['brand']}, {doc['subcategory']})")
    
