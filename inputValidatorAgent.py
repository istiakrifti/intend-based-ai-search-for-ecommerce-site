from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class InputValidator:
    def __init__(self, question, context):
        self.question = question
        self.context = context
    
    def validate(self):
        template="""
            You are a helpful assistant for question-answering task.
            You are given a question and a context. Your task is to determine if the question is relevant to the context.
            If the question is relevant to the context, respond with 'Yes'. If it is not relevant, respond with 'No'.
            Do not provide any additional information or explanation. 
            Question: {question} 
            Context: {context}
            Answer:
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

    
        result = rag_chain.invoke({"question": self.question, "context": self.context})

        return result['text']
    