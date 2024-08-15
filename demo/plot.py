import os
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.docstore.document import Document
from typing import List
from documentation_site.md_loader import MarkDownLoader
import time
import openai


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"


load_dotenv()
_ = load_dotenv(find_dotenv())  # read local .env file

#Azure_OpenAI llm    
llm_azure = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], 
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_4"],
    deployment_name=os.environ["DEPLOYMENT_NAME_4"],
    openai_api_type=os.environ["AZURE_OPENAI_API_TYPE"]
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["DEPLOYMENT_NAME_EMBEDDING"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_4"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)


loader = MarkDownLoader('demo/data_analysis.md')
data = loader.load()
print(data[0])
print(len(data))

# vectordb = Chroma.from_documents(
#     documents=data,
#     embedding=embeddings,
#     persist_directory="demo/vectordb/chroma/data_analysis"
# )
# vectordb.persist()
vectordb = Chroma(persist_directory="demo/vectordb/chroma/data_analysis", embedding_function=embeddings)
retriever = vectordb.as_retriever(k=3)

# class APIRetriever(BaseRetriever):
#     """ A Openapi retriever to return the entire content from api call, 
#         always pick a keyword from below question
#     """ 
#     document: List[Document]

#     def _get_relevant_documents(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> str:
#         return self.document


ANSWER_PROMPT = """You are a data visualization expert
with a strong background in matplotlib.
Use the context and your knowledge to answer
return the code of generating plot to user


Question: {question}
context: {context}
"""
prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)
chain = (
       {"context": retriever, "question": RunnablePassthrough()} 
        | prompt
        | llm_azure
        | StrOutputParser())


def getAnswer(message):
    print(message)
    return chain.invoke(message)


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    
    history_langchain_format.append(HumanMessage(content=message))
    
    try:
        gpt_response = getAnswer(message)

    except openai.BadRequestError as e:
        gpt_response = "Sorry, no keywords of study found. Please try again with keywords that applies to name and cancer type of the studies" 
   
    except openai.APIConnectionError as e:
        gpt_response = "Server connection error: {}".format(e)

    except Exception as e:
        gpt_response = "Error: {}".format(e)

    # chat_history = chat_history[:-1]    
    partial_message = ""

    for i in range(len(gpt_response)):
        partial_message += gpt_response[i]
        time.sleep(0.01)
        yield partial_message    


chatbot = gr.Chatbot(  # uploaded image of user and cBioportal as avatar 
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=("demo/sample_data/user_avatar.png", 
                    "demo/sample_data/chatbot_avatar.png"),
    )
gr.ChatInterface(predict, chatbot=chatbot, title="cBioPortal plot ChatBot").launch()
