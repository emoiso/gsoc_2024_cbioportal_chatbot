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
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_NAME"],
    openai_api_type=os.environ["OPENAI_API_TYPE"]
)

openapi_prompt = """ 
        Select projection of DETAILED to call the api ,
        if user mentioned meta, summary or ID , then use that for projection instead.
        if you get nothing in api call, just say you didn't find anything.

        "{question}"
        """
prompt = PromptTemplate(template=openapi_prompt, input_variables=["question"])

open_api_spec = OpenAPISpec.from_file("/Users/xinling/Desktop/cBio/Chatbot/langchain_gsoc_main/cBioPortal_openapi_3.yaml")

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
You are a professional assistant, manage the retrieved data to human language and return it to users,
if response is empty, just say you didn't find anything
Question : {question}
{context}
"""
prompt_chatbot = ChatPromptTemplate.from_template(answer_prompt)
# build a chain # lambda x:


def get_response(question):
    openapi_response = call_openapi(question)
    # retriever = RunnableParallel(chain_cBio(question))
    chain = (
            {"question": RunnablePassthrough(), "context": APIRetriever(document=openapi_response)}
            | prompt_chatbot  
            | llm_azure
            | StrOutputParser()
    )
    ans = chain.invoke(question)
    return ans

# ask_question("which studies have treated samples?")
# which studies have patient-derived xenografts PDXs?
# which studies were conducted in NY?

# User Interface
def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = get_response(message)
    return gpt_response

chatbot = gr.Chatbot(  # uploaded image of user and cBioportal as avatar 
    [],
    elem_id="chatbot",
    bubble_full_width=False,
    avatar_images=((os.path.join(os.path.dirname(__file__), "sample_data/user_avatar.png")), 
                    (os.path.join(os.path.dirname(__file__), "sample_data/chatbot_avatar.png"))),
)

gr.ChatInterface(predict, title="cBioPortal OPENAPI ChatBot", chatbot=chatbot).launch(share=True)