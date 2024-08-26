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
from plot import get_plot_chain
import time
import openai
from pathlib import Path
import time
from datetime import datetime
import tempfile


load_dotenv()
_ = load_dotenv(find_dotenv())  # read local .env file

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"


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


classify_chain = (ChatPromptTemplate.from_template(
    """Given the user question and description below, classify it as either being about 
    'pmc', 'documentation', 'issue', 'study', 'plot', or 'all_chains'. 
    
    Use the chain if user mention it, like use all_chains
    pmc: including PMC papers. Choose this if user query mention paper title or anything related to paper
    documentation: including intro and user guide for cBioportal
    issue:  including discussions of issues while using cBioportal
    all_chains: including all info from other topics. Choose it if u are not sure which topic fits.
    study: get study and sample data by studyID, cancer name, sampleID.
    plot: write code to make plots
    
    Do not respond with more than one word.
    Question: {question}  
    Chat_history: {chat_history}                  
    Classification:""") | llm_azure | StrOutputParser())


pmc_chain = get_pubmed_chain()
mbox_chain = get_mbox_chain()
documentation_site_chain = get_documentation_site_chain()
all_chain = get_all_chain()
plot_chain = get_plot_chain()
chain_info = ""
second_call = False


def route(info):
    global chain_info, second_call
    print(info)
    chain_info = info['topic']
    if second_call:
        second_call = False
        chain_info = "all_chains"
        return all_chain
    if "issue" in info['topic'].lower():
        return mbox_chain

    elif "pubmed" in info['topic'].lower(): 
        return pmc_chain

    elif "documentation" in info['topic'].lower():
        return documentation_site_chain

    elif "study" in info['topic'].lower():
        response = call_openapi(info['question'])
        chain = get_openapi_chain(response)
        return chain

    elif "plot" in info['topic'].lower():
        return plot_chain
    else:
        return all_chain


def initialize_chat_history_file():
    global chat_history_file
    filename = "chat_history.txt"
    temp_dir = 'saved_chats'  # Use temp directory
    current_path = os.getcwd()
    chat_history_file = current_path +"/" + temp_dir +"/" + filename
    with open(chat_history_file, 'w') as file:
        pass 
    print(f"Chat history will be saved to: {chat_history_file}")


def save_chat_history():
    global chat_history_file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(chat_history_file)
    with open(chat_history_file, "a") as f:
        for i in range(0, len(chat_history), 2):
            user_message = chat_history[i]
            ai_response = chat_history[i + 1] if i + 1 < len(chat_history) else ""
            f.write(f"({timestamp})\n")
            f.write(f"User: {user_message}\n")
            f.write(f"AI: {ai_response}\n")
        f.close()    
    # Check if the file exists and print its path
    if os.path.exists(chat_history_file):
        print(f"File exists: {chat_history_file}, Size: {os.path.getsize(chat_history_file)} bytes")
    else:
        print(f"File not found: {chat_history_file}")

    return str(chat_history_file)


chat_history = []


def getAnswer(question):
    ans = ({'topic': classify_chain, "question": lambda x: x} | RunnableLambda(route))
    return ans.invoke({"question": question, "chat_history": chat_history})


# User Interface
def predict(message, history):
    global chat_history, second_call 
    history_langchain_format = []
    
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    try:
        gpt_response = getAnswer(message)
        if chain_info == "study" and "I didn't find anything" in gpt_response:
            second_call = True
            gpt_response = getAnswer(message)

        chat_history.extend([HumanMessage(content=message), gpt_response])

    except openai.BadRequestError as e:
        gpt_response = "Sorry, no keywords of study found. Please try again with keywords that applies to name and cancer type of the studies" 
   
    except openai.APIConnectionError as e:
        gpt_response = "Server connection error: {}".format(e)

    except Exception as e:
        gpt_response = "Error: {}".format(e)
        print(e)

    # Save chat history after every interaction
    saved_file = save_chat_history()
    print(f"Chat history saved to: {saved_file}")
    time.sleep(0.1)

    gr.DownloadButton(value=str(saved_file))
    
    partial_message = "Source Document Type : " + chain_info + "\n"
    
    for i in range(len(gpt_response)):
        partial_message += gpt_response[i]
        time.sleep(0.01)
        yield partial_message   

    chat_history = chat_history[-2:] #only keep latest conversation, old messages are removed


chatbot = gr.Chatbot(  # uploaded image of user and cBioportal as avatar 
    [],
    elem_id="chatbot",
    bubble_full_width=False,
    avatar_images=("demo/sample_data/user_avatar.png", 
                   "demo/sample_data/chatbot_avatar.png"),
)

initialize_chat_history_file()

with gr.Blocks() as demo:
    gr.ChatInterface(
                    predict, 
                    title="cBioPortal ChatBot",
                    # css="""footer{display:none !important}""",
                    examples=['How to install cBioportal in docker', 'what are important gene list in "Comprehensive molecular portraits of human breast tumors"', 
                    'Give me some studies related to bone', 'can u give me some samples related to TCGA-OR-A5J1-01', 'Tell me a joke', 'how to make a plot of the top 10 most mutated genes group by hugoGeneSymbol'],
                    chatbot=chatbot)
    
    gr.HTML(f"""
        <a href= "/file={str(chat_history_file)}" download>
            <button style="
                background-color: rgba(51, 51, 51, 0.05);
                border-radius: 8px;
                border-width: 0;
                color: #333333;
                width: 100%;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 20px auto; /* Centers the button */
                cursor: pointer;
                border-radius: 5px;
            "
            >Download the conversation</button>
        </a>
    """),
    
    myfooter = gr.HTML("""
                    <footer class="disclaimer">
                        <p style="white-space: pre-line;color: #9E9E9E">\n\nThis chatbot can make mistakes. Check important info.</p>
                    </footer>
                    <style>
                        .disclaimer {
                            bottom: 0;
                            width: 100%;
                            margin-top:auto;
                            padding: 0.5rem;
                            text-align: center;
                            font-size: 1rem;
                            position: relative;
                        }
                    </style>"""
                       )
 
demo.launch(allowed_paths=[str(chat_history_file)],
            share=True)
