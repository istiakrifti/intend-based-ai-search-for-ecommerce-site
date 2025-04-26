from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class InputValidator:
    def __init__(self, question):
        self.question = question
        # self.context = context
    
    def validate(self):
        template = """
        We sell products in these categories: laptops, phones, TVs, accessories, and smartwatches.

        You are a helpful assistant. If the user's question is about any of the categories we sell, respond with "Yes", otherwise respond with "No".

        Question: {question}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        prompt.pretty_print()

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        rag_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=StrOutputParser(),
            # verbose=True
        )

        result = rag_chain.invoke({"question": self.question})

        return result['text']
    