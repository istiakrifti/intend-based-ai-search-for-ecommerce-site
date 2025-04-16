from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class InputValidator:
    def __init__(self, question, context):
        self.question = question
        self.context = context
    
    def validate(self):
        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            You are a helpful assistant for question-answering task.
            You are given a question and a context. Your task is to determine if the question is relevant to the context.
            If the question is relevant to the context, respond with 'Yes'. If it is not relevant, respond with 'No'.
            Do not provide any additional information or explanation. 
            Question: {question} 
            Context: {context}
            Answer:
            """
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        rag_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=StrOutputParser(),
            # verbose=True
        )

    
        result = rag_chain.invoke({"question": self.question, "context": self.context})

        return result['text']
    

# if __name__ == "__main__":
#     question = "What is the capital of Bangladesh?"
#     context = "The capital of France is Paris."
    
#     validator = InputValidator(question, context)
#     result = validator.validate()
    
#     print(result)