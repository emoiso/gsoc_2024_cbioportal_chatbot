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
# from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import json


#langchain-core==0.2.3


def load_json_file(path):
    with open(path) as f:
        data = json.load(f)
    return data  


pmid_data = load_json_file('pubmed/pmcid_list.json')
study_data = load_json_file('pubmed/cBioportal_study.json')
pmid_dict = {}
study_dict = {}

# pair pmid and pmcid in dict
for data in pmid_data:
    pmcid = data.get('pmcid')
    if pmcid:
        pmid = data.get('pmid')
        pmid_dict[pmcid] = pmid

# get and save each study content 
for study in study_data:
    pmid = study.get('pmid')
    if pmid:
        content_dict = {}
        content_dict['name'] = study.get('name')
        content_dict['description'] = study.get('description')
        content_dict['publicStudy'] = study.get('publicStudy')
        content_dict['citation'] = study.get('citation')
        content_dict['groups'] = study.get('groups')
        content_dict['status'] = study.get('status')
        content_dict['importDate'] = study.get('importDate')
        content_dict['allSampleCount'] = study.get('allSampleCount')
        content_dict['readPermission'] = study.get('readPermission')
        content_dict['studyId'] = study.get('studyId')
        content_dict['cancerTypeId'] = study.get('cancerTypeId')
        content_dict['referenceGenome'] = study.get('referenceGenome')
    study_dict[pmid] = content_dict    


class pubmedLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],

    ):
        self.file_path = Path(file_path).resolve()

    def load(self) -> List[Document]:  
        """Load and return documents from the pubmed file."""
        docs: List[Document] = []

        with open(self.file_path, encoding="utf-8") as f:
            pubmed_content = f.read()
        #extract pmcid from file name, and get pmid & study content
        pmcid = Path(self.file_path).stem
        pmid = pmid_dict[pmcid]
        study = {}
        if pmid in study_dict:
            study = study_dict[pmid]

        # extract paper title 
        with open(self.file_path, encoding="utf-8") as file:
            title = file.readline().strip()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512, 
            chunk_overlap=256,
            separators=["\n", "Results", "Discussion", "Method Summary", "Tissue Source Sites", "Supplementary Material"]
        )        
        pubmed_splits = text_splitter.split_text(pubmed_content)
        #convert string to doc, otherwise it cannot have metaData
        pubmed_splits_doc = text_splitter.create_documents(pubmed_splits)
        for split in pubmed_splits_doc:
            docs.append(split)
        # add metaData
        for doc in docs:
            doc.metadata['pmc_id'] = pmcid
            doc.metadata['paper_title'] = title
            doc.metadata['pmid'] = pmid
            if study:
                doc.metadata['name'] = study['name']
                doc.metadata['description'] = study['description']
                doc.metadata['publicStudy'] = study['publicStudy']
                doc.metadata['citation'] = study['citation']
                doc.metadata['groups'] = study['groups']
                doc.metadata['status'] = study['status']
                doc.metadata['importDate'] = study['importDate']
                doc.metadata['allSampleCount'] = study['allSampleCount']
                doc.metadata['readPermission'] = study['readPermission']
                doc.metadata['studyId'] = study['studyId']
                doc.metadata['cancerTypeId'] = study['cancerTypeId']
                doc.metadata['referenceGenome'] = study['referenceGenome']
        return docs


# loader = DirectoryLoader('pubmed/test_chatbot', 
#                         loader_cls=pubmedLoader, show_progress=True)

# # loader = pubmedLoader('loaded_pmc/PMC2671642.txt')
# # load and split
# docs = loader.load()  

# with open('pubmed_test_oldMetadata.txt', 'w') as f:
#     f.write(str(docs))


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

# vectordb = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     persist_directory="vectordb/chroma/pubmed/test"
# )

# vectordb.persist()
# retriever = vectordb.as_retriever(k=5)


#reuse the vectordb created before
vectordb = Chroma(persist_directory="vectordb/chroma/markdown", embedding_function=embeddings)   
# print(vectordb.get())


# self-query retriever
document_content_description = "full text of Pubmed papers "
metadata_field_info = [
    AttributeInfo(
        name="pmc_id",
        description="The id for pubmed paper, each id is unique",
        type="string",
    ),
    AttributeInfo(
        name="title",
        description="The title for pubmed paper",
        type="string",
    ),
    AttributeInfo(
        name="name",
        description="The name of study using this pubmed paper",
        type="string",
    ),
    AttributeInfo(
        name="description",
        description="The description of study using this pubmed paper",
        type="string",
    ),
    AttributeInfo(
        name="publicStudy",
        description=" Whether the study is public",
        type="boolean",
    ),
    AttributeInfo(
        name="citation",
        description="The citation of study using this pubmed paper",
        type="string",
    ),
    AttributeInfo(
        name="groups",
        description="The groups of study using this pubmed paper",
        type="string",
    ),
    AttributeInfo(
        name="status",
        description="The status of study using this pubmed paper",
        type="int",
    ),
    AttributeInfo(
        name="importDate",
        description="The importDate of study using this pubmed paper",
        type="date",
    ),
    AttributeInfo(
        name="allSampleCount",
        description="The allSampleCount of study using this pubmed paper",
        type="int",
    ),
    AttributeInfo(
        name="readPermission",
        description="The readPermission of study using this pubmed paper",
        type="boolean",
    ),
    AttributeInfo(
        name="studyId",
        description="The studyId of study using this pubmed paper",
        type="string",
    ),
    AttributeInfo(
        name="cancerTypeId",
        description="The cancerTypeId of study using this pubmed paper",
        type="string",
    ),
    AttributeInfo(
        name="referenceGenome",
        description="The referenceGenome of study using this pubmed paper",
        type="string",
    ),
    AttributeInfo(
        name="name",
        description="The name of study using this pubmed paper",
        type="string",
    ),
]
SelfQuery_Retriever = SelfQueryRetriever.from_llm(
    llm_azure,
    vectordb,
    document_content_description,
    metadata_field_info
)


# build prompt template
ANSWER_PROMPT = """You are a professional assistant. Answer questions using content and metadata.
                Also, ignore the context if it is a reference.
                Answer the question based only on the following context, 
                and return the pmc_id, pmid, studyID of all the contexts :
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)
# build a chain # lambda x:
chain = (
    {"context": SelfQuery_Retriever, "question": RunnablePassthrough()} 
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
    avatar_images=("sample_data/user_avatar.png", 
                   "sample_data/chatbot_avatar.png"),
)

gr.ChatInterface(predict, title="cBioPortal pubmed ChatBot", chatbot=chatbot).launch()