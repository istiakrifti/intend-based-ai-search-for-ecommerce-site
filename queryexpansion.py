from typing import List, Optional, Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class QueryExpansionEngine:
    """
    A dedicated class for expanding queries using LLMs.
    This class generates multiple variations of a query to improve retrieval results.
    """
    
    def __init__(self, 
                llm=None, 
                num_queries: int = 4,
                model_name: str = None,
                temperature: float = 0.5):
        """
        Args:
            llm: LLM to use for query expansion (if None, will initialize based on model_name)
            num_queries: Number of expanded queries to generate
            model_name: Model name to use if llm not provided
            temperature: Temperature for generation
        """
        self.num_queries = num_queries
        
        # Initialize LLM if not provided
        if llm is None:
            self._initialize_llm(model_name, temperature)
        else:
            self.llm = llm
            
        # Create the query expansion chain
        self._create_expansion_chain()
    
    def _initialize_llm(self, model_name: Optional[str], temperature: float):
        """Initialize LLM based on available providers"""
        if model_name is None:
            # Default to Groq if no model specified
            try:
                self.llm = ChatGroq(temperature=temperature)
                return
            except ImportError:
                pass
        
        # Try different providers based on model name or what's available
        if model_name is None or "gpt" in model_name.lower():
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model_name=model_name or "gpt-3.5-turbo", 
                    temperature=temperature
                )
                return
            except ImportError:
                pass
                
        if model_name is None or "claude" in model_name.lower():
            try:
                self.llm = ChatAnthropic(
                    model_name=model_name or "claude-instant-1", 
                    temperature=temperature
                )
                return
            except ImportError:
                pass
        
        # Use HuggingFace models as fallback
        try:
            from langchain_community.llms import HuggingFaceHub
            self.llm = HuggingFaceHub(
                repo_id=model_name or "google/flan-t5-small",
                model_kwargs={"temperature": temperature}
            )
            return
        except ImportError:
            pass
            
        raise ValueError("No supported LLM found. Please install langchain-openai, langchain-anthropic, or langchain-groq")
    
    def _create_expansion_chain(self):
        """Create the query expansion chain"""
        template = f"""You are a helpful assistant that generates multiple search queries based on a single input query.
Your task is to create diverse variations that capture different aspects and phrasings of the original query.
Generate {self.num_queries} different search queries related to: {{question}}

Output exactly {self.num_queries} queries, one per line. Do not include numbering or any other text:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        self.expansion_chain = (
            prompt 
            | self.llm
            | StrOutputParser() 
            | (lambda x: [q.strip() for q in x.strip().split("\n") if q.strip()])
        )
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query into multiple variations
        
        Args:
            query: The original query to expand
            
        Returns:
            List of expanded queries
        """
        try:
            expanded_queries = self.expansion_chain.invoke({"question": query})
            
            # Ensure we have the requested number of queries
            if len(expanded_queries) < self.num_queries:
                # Add the original query if not enough expansions
                if query not in expanded_queries:
                    expanded_queries.append(query)
                
                # # Still not enough? Duplicate some with small modifications
                # while len(expanded_queries) < self.num_queries:
                #     idx = len(expanded_queries) % len(expanded_queries)
                #     expanded_queries.append(f"about {expanded_queries[idx]}")
            
            # Ensure original query is included
            if query not in expanded_queries:
                expanded_queries[0] = query
                
            return expanded_queries
            
        except Exception as e:
            print(f"Query expansion failed: {str(e)}")
            # Fallback: return original query
            return [query]
    
    def expand_batch(self, queries: List[str]) -> Dict[str, List[str]]:
        """
        Expand multiple queries in batch
        
        Args:
            queries: List of queries to expand
            
        Returns:
            Dictionary mapping original queries to their expansions
        """
        results = {}
        for query in queries:
            results[query] = self.expand_query(query)
        return results