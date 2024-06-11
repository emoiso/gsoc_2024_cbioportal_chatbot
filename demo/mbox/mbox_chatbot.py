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


#langchain-core==0.2.3
def main():
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
    loader = DirectoryLoader("mbox", glob='**/*.txt', show_progress=True)
    raw_documents = loader.load()

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["DEPLOYMENT_NAME_EMBEDDING"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"]
    )
    vectordb = Chroma.from_documents(
        documents=raw_documents,
        embedding=embeddings,
        persist_directory="vectordb/chroma/mbox_fivemessage"
    )
    vectordb.persist()

    retriever = vectordb.as_retriever(k=2)
    # build prompt template
    ANSWER_PROMPT = """Answer the question based only on the following context, 
                    and return the url of all the contexts :
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

    gr.ChatInterface(predict, title="cBioPortal Mbox ChatBot", chatbot=chatbot).launch()

if __name__ == '__main__':
    main()
