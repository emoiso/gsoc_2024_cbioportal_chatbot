from langchain_community.document_loaders import DirectoryLoader
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
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()
_ = load_dotenv(find_dotenv())  # read local .env file


def load_and_split(path):
    loader = JSONLoader(
        file_path=path,
        jq_schema=".",
        text_content=False
    )

    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["},", '\n'],
        chunk_size=512,
        chunk_overlap=200)
    split_docs = text_splitter.split_documents(data)
    return split_docs


# split_docs = load_and_split('demo/mbox/deleted_error_mbox.json')
# docs = []
# for doc in split_docs:
#     doc.metadata.clear()
#     docs.append(doc)
# print(len(docs))


# def write_json_file(fileName, data):
#     with open(fileName, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)


# write_json_file('deleted_error_mbox.txt', str(docs))

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


# # first part of embeddings
# vectordb = Chroma.from_documents(
#     documents=docs[:2],
#     embedding=embeddings,
#     persist_directory="demo/vectordb/chroma/mbox_07"
# )
# vectordb.persist()


def update_vectordb_with_docs(docs, embeddings, base_persist_directory):
    # the db created for first part of db
    vectordb_total = Chroma(persist_directory=base_persist_directory, embedding_function=embeddings) 
    # Iterate through the documents and update the vectordb
    for i in range(len(docs) // 2):
        start, end = 2+i * 2, 2+ 2 * (i + 1)
        docs_to_embed = docs[start:end]
        try:
            vectordb_new = Chroma.from_documents(
                documents=docs_to_embed,
                embedding=embeddings
            )
            
            new_data = vectordb_new._collection.get(include=['documents', 'metadatas', 'embeddings'])
            vectordb_total._collection.add(
                embeddings=new_data['embeddings'],
                metadatas=new_data['metadatas'],
                documents=new_data['documents'],
                ids=new_data['ids']
            )
        except Exception as e:
            print(e)
            print(str(i) + " is too big")
        print(vectordb_total._collection.count())  
    return vectordb_total


# vectordb = update_vectordb_with_docs(docs, embeddings, "demo/vectordb/chroma/mbox")
vectordb = Chroma(persist_directory="demo/vectordb/chroma/mbox_07", embedding_function=embeddings)


# self-query retriever
document_content_description = "conversations about cBioportal "
metadata_field_info = [
    AttributeInfo(
        name="url",
        description="The url for google group conversation",
        type="string",
    )
]
SelfQuery_Retriever = SelfQueryRetriever.from_llm(
    llm_azure,
    vectordb,
    document_content_description,
    metadata_field_info,
    search_kwargs={'k': 3}
    # search_type="mmr"
)
retriever = vectordb.as_retriever(k=3)

# build prompt template
ANSWER_PROMPT = """
                You are a professional assistant working for cBioportal, 
                Answer the question based only on the following context and chat history if needed, 
                Below is a set of related Q&A examples that includes both good and bad examples. For each example:
                If it is marked as a 'Good example,' you may refer the conversation. Sometimes user can give important info.
                If it is marked as a 'Bad example,' improve the answer to better meet the user's needs.
                Return the url of conversation (should inside contexts) :

                Do not mention user name from conversations.
{related_QA}
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)


def get_mbox_chain():
    chain = (
        RunnableLambda(lambda x: x['question']) |
        {"related_QA": RunnablePassthrough(), "context": SelfQuery_Retriever, "question": RunnablePassthrough()} 
        | prompt  # choose a prompt
        | llm_azure  # choose a llm
        | StrOutputParser()
    )
    return chain


def main():
    def getAnswer(question):
        chain = get_mbox_chain
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
        avatar_images=("demo/sample_data/user_avatar.png", 
                    "demo/sample_data/chatbot_avatar.png"),
    )

    gr.ChatInterface(predict, title="cBioPortal Mbox ChatBot", chatbot=chatbot).launch()


if __name__ == '__main__':
    main.run()
