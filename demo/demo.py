import codecs
import json
from llama_index.readers.mbox import MboxReader
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
import os
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import gradio as gr
from util.loader import MboxLoader, JSONLoader, MarkDownLoader
from langchain.schema import AIMessage, HumanMessage


def get_docs(doc_type):
    if doc_type == "mbox":
        pass
    elif doc_type == "markdown":
        pass
    elif doc_type == "json":
        pass
    else:
        return []


if __name__ == '__main__':
    # change target data type
    # json, md, mbox
    DATA_TYPE = "md"

    # replace your config.json
    # load config
    with codecs.open("config.json", encoding="utf-8") as f:
        config = json.load(f)
        if "OPENAI_BASE_URL" in config:
            os.environ["OPENAI_BASE_URL"] = config["OPENAI_BASE_URL"]

        OPENAI_API_KEY = config["OPENAI_API_KEY"]
        ELASTIC_CLOUD_ID = config["ELASTIC_CLOUD_ID"]
        ES_USER = config["ES_USER"]
        ES_PASSWORD = config["ES_PASSWORD"]


    SAMPLE_DATA_DIR = "sample_data/"

    DATA_PATH = SAMPLE_DATA_DIR + "/sample." + DATA_TYPE
    INDEX_NAME = "sample_" + DATA_TYPE + "_index"

    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>offline start<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

    # choose embedding model
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_store = ElasticsearchStore(
        es_cloud_id=ELASTIC_CLOUD_ID,
        es_user=ES_USER,
        es_password=ES_PASSWORD,
        index_name=INDEX_NAME,
        embedding=embedding_model,
    )

    # choose different loader
    if DATA_TYPE == "json":
        def demo_metadata_func(record: dict, metadata: dict) -> dict:
            metadata["name"] = record.get("name")
            metadata["summary"] = record.get("summary")
            metadata["url"] = record.get("url")
            metadata["category"] = record.get("category")
            metadata["updated_at"] = record.get("updated_at")

            return metadata
        loader = JSONLoader(DATA_PATH, "content", metadata_func=demo_metadata_func)
    elif DATA_TYPE == "md":
        loader = MarkDownLoader(DATA_PATH)
    elif DATA_TYPE == "mbox":
        loader = MboxLoader(DATA_PATH)
    else:
        loader = UnstructuredFileLoader(DATA_PATH) 

    # split page_content into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=256
    )
    raw_documents = loader.load_and_split(text_splitter=text_splitter)
    for doc in raw_documents:
        print(doc)
    # add embeddings into documents
    # save documents into elasticsearch
    documents = vector_store.from_documents(
        raw_documents,
        embedding_model,
        index_name=INDEX_NAME,
        es_cloud_id=ELASTIC_CLOUD_ID,
        es_user=ES_USER,
        es_password=ES_PASSWORD
    )

    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>offline end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>online start<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    
    # online query mode
    retriever = vector_store.as_retriever()

    # choose llm model
    llm_model = OpenAI(openai_api_key=OPENAI_API_KEY)

    # build prompt template
    ANSWER_PROMPT = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as verbose and educational in your response as possible.

        context: {context}
        Question: "{question}"
        Answer:
        """
    )

    # build chain
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | ANSWER_PROMPT
            | llm_model
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

    chatbot = gr.Chatbot( #uploaded image of user and cBioportal as avatar 
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=((os.path.join(os.path.dirname(__file__), "sample_data/user_avatar.png")), (os.path.join(os.path.dirname(__file__), "sample_data/chatbot_avatar.png"))),
    )

    gr.ChatInterface(predict, title="cBioPortal ChatBot", chatbot=chatbot).launch()


">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>online end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"