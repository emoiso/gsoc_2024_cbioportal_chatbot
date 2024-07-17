 from langchain.document_loaders import DirectoryLoader
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
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


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
#     documents=raw_documents,
#     embedding=embeddings,
#     persist_directory="vectordb/chroma/markdown"
# )
# vectordb.persist()
vectordb = Chroma(persist_directory="demo/vectordb/chroma/documentation_site", embedding_function=embeddings)

# self-query retriever
document_content_description = "documentation site of cBioportal "
metadata_field_info = [
    AttributeInfo(
        name="url",
        description="The url for document_content",
        type="string",
    )
]
SelfQuery_Retriever = SelfQueryRetriever.from_llm(
    llm_azure,
    vectordb,
    document_content_description,
    metadata_field_info
)

# retriever = vectordb.as_retriever(k=3)
# build prompt template
ANSWER_PROMPT = """Answer the question based only on the following context, 
                and return the url of all the contexts :
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)


def get_documentation_site_chain():
    chain = (
        {"context": SelfQuery_Retriever, "question": RunnablePassthrough()} 
        | prompt  # choose a prompt
        | llm_azure  # choose a llm
        | StrOutputParser()
    )
    return chain


def getAnswer(question):
    chain = get_documentation_site_chain()
    ans = chain.invoke(question)
    return ans


# # User Interface
# def predict(message, history):
#     history_langchain_format = []
#     for human, ai in history:
#         history_langchain_format.append(HumanMessage(content=human))
#         history_langchain_format.append(AIMessage(content=ai))
#     history_langchain_format.append(HumanMessage(content=message))
#     gpt_response = getAnswer(message)
#     return gpt_response


# chatbot = gr.Chatbot(  # uploaded image of user and cBioportal as avatar 
#     [],
#     elem_id="chatbot",
#     bubble_full_width=False,
#     avatar_images=("demo/sample_data/user_avatar.png", 
#                    "demo/sample_data/chatbot_avatar.png"),
# )

# gr.ChatInterface(predict, title="cBioPortal Markdown ChatBot", chatbot=chatbot).launch()
