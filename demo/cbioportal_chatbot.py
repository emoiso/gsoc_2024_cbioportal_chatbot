import os
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from openapi.cBioPortal_api_demo import get_openapi_chain, call_openapi
from pubmed.pubmed_demo import get_pubmed_chain
from mbox.mbox_chatbot import get_mbox_chain
from documentation_site.markdown_demo import get_documentation_site_chain
from all import get_all_chain
import time
import openai


load_dotenv()
_ = load_dotenv(find_dotenv())  # read local .env file

#Azure_OpenAI llm    
llm_azure = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], 
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_NAME"],
    openai_api_type=os.environ["OPENAI_API_TYPE"]
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["DEPLOYMENT_NAME_EMBEDDING"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)


classify_chain = (ChatPromptTemplate.from_template(
    """Given the user question and description below, classify it as either being about 'pmc', 'documentation',
     'study' or 'issue', or 'other'. 

    pmc: including PMC papers in Cancer Genomics domain
    issue:  including discussions of issues while using cBioportal
    documentation: including intro and user guide for cBioportal
    study: studies in cbioportal
    other: cannot be classify by words above or need more than one words to classify

    Do not respond with more than one word.
    Question: {question}                    
    Classification:""") | llm_azure | StrOutputParser())


pmc_chain = get_pubmed_chain()
mbox_chain = get_mbox_chain()
documentation_site_chain = get_documentation_site_chain()
all_chain = get_all_chain()


def route(info):
    print(info)
    if "issue" in info['topic'].lower():
        return mbox_chain

    elif "pubmed" in info['topic'].lower():  # good
        return pmc_chain

    elif "documentation" in info['topic'].lower():
        return documentation_site_chain

    elif "study" in info['topic'].lower():   # good
        response = call_openapi(info['question'])
        chain = get_openapi_chain(response)
        return chain

    else:
        return all_chain


def getAnswer(question):
    ans = {'topic': classify_chain, "question": lambda x: x} | RunnableLambda(route)
    return ans.invoke(question)


# User Interface
def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    try:
        gpt_response = getAnswer(message)
    except openai.BadRequestError as e:
        gpt_response = "Sorry, no study found. Please try again with keywords that applies to name and cancer type of the studies" 
   
    except openai.APIConnectionError as e:
        gpt_response = ("Server connection error: {e.__cause__}") 

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

gr.ChatInterface(
                predict,
                title="cBioPortal ChatBot", 
                examples=['How to install cBioportal in docker', 'Can u summarize pmc_id PMC2671642?', 
                          'Give me some studies related to bone', 'can u give me some samples related to TCGA-OR-A5J1-01', 'Issue about uploading-Analyzing public/personal data','Does cbioportal taste like chocolate'],
                chatbot=chatbot).launch()
