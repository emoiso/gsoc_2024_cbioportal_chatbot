from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.openai_functions.openapi import get_openapi_chain
from langchain_community.utilities.openapi import OpenAPISpec
#from langchain_openai import AzureChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough 
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


#Pip install langchain==0.0.343
#langchain-core-0.0.13
def main():
    # """ Example for OPENAPI chain"""
    # chain_klarna = get_openapi_chain(
    #     spec="https://www.klarna.com/us/shopping/public/openai/v0/api-docs/",
    #     verbose=True
    # )
    # print(chain_klarna("What are some options for a men's large blue button down shirt"))

    load_dotenv()
    _ = load_dotenv(find_dotenv())  # read local .env file

    llm_azure = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], 
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        deployment_name=os.environ["DEPLOYMENT_NAME"],
        openai_api_type=os.environ["OPENAI_API_TYPE"]
    )

    api_prompt = """ 
            You are a professional assistant, Use the provided API's to respond to this user query: 
            select projection of DETAILED to call the api ,
            if user mentioned meta, summary or ID , then use that for projection instead.
            if you get nothing in api call, just say you didn't find anything.

            "{question}"
            """
    prompt = PromptTemplate(template=api_prompt, input_variables=["question"])
    
    open_api_spec = OpenAPISpec.from_file("/Users/xinling/Desktop/cBio/Chatbot/langchain_gsoc_main/cBioPortal_openapi_3.yaml")

    chain_cBio = get_openapi_chain(
        spec=open_api_spec,
        llm=llm_azure,
        prompt=prompt,
        verbose=True
    )

    def ask_question(question: str):
        ans = chain_cBio(question)
        # print(ans)
        return str(ans['response'])

    answer_prompt = """
    You are a professional assistant, manage the retrieved data and return it to users,
    if response is empty, just say you didn't find anything
    {context}
    """
    prompt_chatbot = ChatPromptTemplate.from_template(answer_prompt)
    # build a chain # lambda x:
    chain = (
        {"context": ask_question(RunnablePassthrough())}
        | prompt_chatbot  # choose a prompt
        | llm_azure  # choose a llm
        | StrOutputParser()
    )

    def get_response(question):
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
        gpt_response = ask_question(message)
        return gpt_response

    chatbot = gr.Chatbot(  # uploaded image of user and cBioportal as avatar 
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=((os.path.join(os.path.dirname(__file__), "sample_data/user_avatar.png")), 
                       (os.path.join(os.path.dirname(__file__), "sample_data/chatbot_avatar.png"))),
    )

    gr.ChatInterface(predict, title="cBioPortal OPENAPI ChatBot", chatbot=chatbot).launch(share=True)


if __name__ == '__main__':
    main()
