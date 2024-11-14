from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains.openai_functions.openapi import get_openapi_chain
from langchain_community.utilities.openapi import OpenAPISpec
#from langchain_openai import AzureChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel 
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


load_dotenv()
_ = load_dotenv(find_dotenv())  # read local .env file

llm_azure = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], 
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_4"],
    deployment_name=os.environ["DEPLOYMENT_NAME_4"],
    openai_api_type=os.environ["AZURE_OPENAI_API_TYPE"]
)


# openapi_prompt = """ 
#         Select projection of DETAILED to call the api ,
#         Have to pick a keyword
#         if you get nothing in api call, just say you didn't find anything.
#         Below is a set of related Q&A examples that includes both good and bad examples. For each example:
#         If it is marked as a 'Good example,' you may refer the conversation. Sometimes user can give important info
#         If it is marked as a 'Bad example,' improve the answer to better meet the user's needs.
#         "{related_QA}"
#         "{question}"
#         """
openapi_prompt = """ 
        Select projection of DETAILED to call the api ,
        Have to pick a keyword
        if you get nothing in api call, just say you didn't find anything.
        "{question}"
        """
prompt = PromptTemplate(template=openapi_prompt, input_variables=["question"])

open_api_spec = OpenAPISpec.from_file("demo/openapi/cBioPortal_openapi_3.yaml")

chain_cBio = get_openapi_chain(
    spec=open_api_spec,
    llm=llm_azure,
    prompt=prompt,
    verbose=True
)


class APIRetriever(BaseRetriever):
    """ A Openapi retriever to return the entire content from api call, 
        always pick a keyword from below question
    """ 
    document: List[Document]

    def _get_relevant_documents(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> str:
        return self.document


def call_openapi(question: str):
    ans = str(chain_cBio(question))
    doc = Document(page_content=ans, metadata={"source": "local"})
    return [doc]


answer_prompt = """
All your job is converting retrieved data to human language 
Make sure the data integrity 
if response is empty, just say you didn't find anything. 

return all data to user and write code to plot if needed
Also, use chat history to response if needed

Below is a set of related Q&A examples that includes both good and bad examples. For each example:
If it is marked as a 'Good response,' you may use it directly if answer user questions
If it is marked as a  'Good instruction', you have to pay attention and follow it.  
If it is marked as a 'Bad example,' improve the answer to better meet the user's needs.
{related_QA}
Question : {question}
{context}
"""
prompt_chatbot = ChatPromptTemplate.from_template(answer_prompt)
# build a chain # lambda x:


def get_openapi_chain(question):
    openapi_response = call_openapi(question)
    chain = (
            {"related_QA": RunnablePassthrough(), "question": RunnablePassthrough(), "context": APIRetriever(document=openapi_response)}
            | prompt_chatbot  
            | llm_azure
            | StrOutputParser()
    )
    return chain


def get_response(question):
    openapi_response = call_openapi(question)
    # retriever = RunnableParallel(chain_cBio(question))
    chain = get_openapi_chain(openapi_response)
    ans = chain.invoke(question)
    return ans


# # User Interface
# def predict(message, history):
#     history_langchain_format = []
#     for human, ai in history:
#         history_langchain_format.append(HumanMessage(content=human))
#         history_langchain_format.append(AIMessage(content=ai))
#     history_langchain_format.append(HumanMessage(content=message))
#     try:
#         gpt_response = get_response(message)
#     except:
#         gpt_response = "Sorry, no study found. Please try again with keywords that applies to name and cancer type of the studies"    
#     return gpt_response


# chatbot = gr.Chatbot(  # uploaded image of user and cBioportal as avatar 
#     [],
#     elem_id="chatbot",
#     bubble_full_width=False,
#     avatar_images=("demo/sample_data/user_avatar.png", 
#                    "demo/sample_data/chatbot_avatar.png")
# )


# gr.ChatInterface(
#                 predict, 
#                 title="cBioPortal OPENAPI ChatBot", 
#                 examples=['Give me some studies related to bone', 'Give me some studies related to tp53'],
#                 chatbot=chatbot
#                 ).launch(show_error=True)