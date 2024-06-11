from langchain.text_splitter import TextSplitter
import requests 
# from indra_nxml_extraction import get_xml, get_xml_from_file, extract_text
from langchain.document_loaders.base import BaseLoader
from pathlib import Path
from typing import Dict, List, Union
from langchain.docstore.document import Document
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser
from typing import Iterator
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document


pmc_id = "PMC6419906"  # BAD IS BAD, BIOINF INTERACT
pmc_id = "PMC3898398"  # GOOD IS GOOD, LUNA SIRT1
#pmc_id = "PMC6070353" # TCGA LUNA/BAD

if pmc_id.upper().startswith('PMC'):
    pmc_id = pmc_id[3:]

pmc_url = 'https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi'

params = {}
params['verb'] = 'GetRecord'
params['identifier'] = 'oai:pubmedcentral.nih.gov:%s' % pmc_id
params['metadataPrefix'] = 'pmc'

# Submit the request
res = requests.get(pmc_url, params)
res.text

pmc_id = "PMC3898398"
# xml_string = get_xml(pmc_id)
# text = extract_text(xml_string)
# # print(text)
# lines = text.splitlines()
# print(lines)

# with open("pubmed_loaded_text.txt", "w") as file:
#     file.write(text)
# docs = List[Document]

# doc = Document(page_content=str(lines), metadata={"pmc_id": pmc_id})
# # docs.append(doc)


def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


# split_doc = get_text_chunks_langchain(text)

class pubmedLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path]
    ):
        self.file_path = Path(file_path).resolve()

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                yield Document(
                    page_content=line,
                    metadata={"pmc_id": pmc_id, "line_number": line_number},
                )
                line_number += 1

    def load(self) -> List[Document]:
        return list(self.lazy_load())


loader = pubmedLoader("pubmed_loaded_text.txt")
docs = loader.load()        
# print(docs)
# split_doc = get_text_chunks_langchain(docs)


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

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="vectordb/chroma/pubmed"
)
vectordb.persist()

retriever = vectordb.as_retriever(k=3)
# build prompt template
ANSWER_PROMPT = """You are a professional assistant.
                Answer the question based only on the following context, 
                and return the pmc_id of all the contexts :
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)
# build a chain # lambda x:
chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt  # choose a prompt
    | llm_azure  # choose a llm
    | StrOutputParser()
)


def getAnswer(question):
    ans = chain.invoke(question)
    return ans


# User Interface
def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = getAnswer(message)
    return gpt_response


chatbot = gr.Chatbot(  # uploaded image of user and cBioportal as avatar 
    [],
    elem_id="chatbot",
    bubble_full_width=False,
    avatar_images=( "sample_data/user_avatar.png", 
                    "sample_data/chatbot_avatar.png"),
)

gr.ChatInterface(predict, title="cBioPortal pubmed ChatBot", chatbot=chatbot).launch()
