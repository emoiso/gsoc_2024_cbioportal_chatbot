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
from openapi.cBioPortal_api_demo import get_openapi_chain, get_response
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
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import traceback
# from langchain_experimental.utilities import PythonREPL


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
# plot_chain = get_plot_chain()
chain_info = ""
second_call = False


def route(info):
    global chain_info, second_call
    print(info)
    chain_info = info['topic']
    if second_call and chain_info == "study":
        second_call = False
        chain_info = "all_chains"
        return all_chain
    # elif second_call and chain_info == "plot":  # second part of plot_chain
    #     second_call = False  # reset the second call
    #     return get_plot_chain(curr_response)   # pass the openapi response

    if "issue" in info['topic'].lower():
        return mbox_chain

    elif "pubmed" in info['topic'].lower(): 
        return pmc_chain

    elif "documentation" in info['topic'].lower():
        return documentation_site_chain

    elif "study" in info['topic'].lower() or "plot" in info['topic'].lower():  # both study and plot call study_chain
        chain = get_openapi_chain(info['question'])
        return chain
    # elif "plot" in info['topic'].lower():
    #     return plot_chain
    else:
        return all_chain


def initialize_chat_history_file():
    global chat_history_file
    filename = "chat_history.txt"
    temp_dir = 'saved_chats'  # Use temp directory
    current_path = os.getcwd()
    chat_history_file = current_path +"/" + temp_dir +"/" + filename
    with open(chat_history_file, 'w') as file: # empty the file when restart this script
        pass 
    print(f"Chat history will be saved to: {chat_history_file}")


def save_chat_history():
    global chat_history_file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(chat_history_file)
    with open(chat_history_file, "a") as f:
        curr_history = chat_history[-2:]
        for i in range(0, len(curr_history), 2):
            user_message = curr_history[i]
            ai_response = curr_history[i + 1] if i + 1 < len(curr_history) else ""
            f.write(f"({timestamp})\n")
            f.write(f"{user_message}\n")
            f.write(f"{ai_response}\n")
        f.close()    
    # Check if the file exists and print its path
    if os.path.exists(chat_history_file):
        print(f"File exists: {chat_history_file}, Size: {os.path.getsize(chat_history_file)} bytes")
    else:
        print(f"File not found: {chat_history_file}")

    return str(chat_history_file)


chat_history = []


def remove_markdown_code_block(markdown_text: str) -> str:
    cleaned_text = markdown_text.replace("```python", "").replace("```", "")
    return cleaned_text.strip()


def generate_docs(conversation):  
    doc = Document(page_content=conversation, metadata={"source": "local"})
    return [doc]


def store_conversation(conversation):
    raw_documents = generate_docs(conversation)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["DEPLOYMENT_NAME_EMBEDDING"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_4"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"]
    )
    if not os.path.isdir("demo/vectordb/chroma/QA"):
        vectordb = Chroma.from_documents(
            documents=raw_documents,
            embedding=embeddings,
            persist_directory="demo/vectordb/chroma/QA"
        )
        vectordb.persist()
    else: 
        # create embedding, then combine
        vectordb_total = Chroma(persist_directory="demo/vectordb/chroma/QA", embedding_function=embeddings)  
        vectordb_new = Chroma.from_documents(
            documents=raw_documents,
            embedding=embeddings,
        )   
        new_data = vectordb_new._collection.get(include=['documents', 'metadatas', 'embeddings'])
        vectordb_total._collection.add(
            embeddings=new_data['embeddings'],
            metadatas=new_data['metadatas'],
            documents=new_data['documents'],
            ids=new_data['ids']
        )
        print(vectordb_total._collection.count()) 
# 把对话存在单独的DB：先call一遍DB再call chain ☑️
# 合并对话&chain DB，对话要被存多次，但是retrieved data 可能会包含QA，也可能把有用的data挤掉


vectordb_QA = Chroma(persist_directory="demo/vectordb/chroma/QA", embedding_function=embeddings)  
retriever_QA = vectordb_QA.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"score_threshold": float(os.getenv("QA_SIMILARITY_THRESHOLD"))}
)


def getAnswer(question):
    # retrieve Q&A data
    related_QA = retriever_QA.invoke(question)
    ans = ({'topic': classify_chain, "question": lambda x: x} | RunnableLambda(route))
    return ans.invoke({"question": question, "chat_history": chat_history, "related_QA": related_QA})


# User Interface
def predict(message, history):
    global chat_history, second_call, curr_response
    history_langchain_format = []
    
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    try:  # call LLM 
        curr_response = getAnswer(message)
        gpt_response = curr_response   

        if chain_info == "plot":
            
            return "Generating a plot..."
            base64_image = gr.Code(value=remove_markdown_code_block(curr_response), language="python")
            image_res = f"![Image Alt Text]({base64_image})"
            gpt_response = image_res

        # if chain_info == "plot" and not second_call:  # got openapi response, now pass it to plot_chain to write code
        #     second_call = True
        #     print("curr_response : ", curr_response)
        #     print("second_call : ", second_call)
        #     gpt_response += getAnswer(message) # for plot chain: return openapi res + plot_chain res

        chat_history.extend([f'User: {HumanMessage(content=message)} ', f'AI: {gpt_response}'])
        # store_conversation([HumanMessage(content=message), gpt_response])
    except openai.BadRequestError as e:
        gpt_response = "Sorry, no keywords of study found. Please try again with keywords that applies to name and cancer type of the studies" 
        print(e)
    except openai.APIConnectionError as e:
        gpt_response = "Server connection error: {}".format(e)
        print(e)
    except Exception as e:
        gpt_response = "Error: {}".format(e)
        print(traceback.format_exc())
        print(e)

    # Save chat history after every interaction
    saved_file = save_chat_history()
    time.sleep(0.1)
    
    partial_message = "Source Document Type : " + chain_info + "\n"
    
    for i in range(len(gpt_response)):
        partial_message += gpt_response[i]
        time.sleep(0.01)
        yield partial_message   

    chat_history = chat_history[-2:]  # only keep latest conversation, old messages are removed


plot_html = """
            <div id="user-browser-plot" style="width:100%;height:400px;"></div>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                // JavaScript code generated by AI for the user's browser to run
                var data = [{
                    x: [1, 2, 3, 4, 5],
                    y: [10, 15, 13, 18, 14],
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: { color: 'blue' }
                }];
                var layout = {
                    title: 'Dynamic Plot Generated by AI',
                    xaxis: { title: 'X-axis Label' },
                    yaxis: { title: 'Y-axis Label' }
                };
                Plotly.newPlot('user-browser-plot', data, layout);
            </script>
            """


def print_like_dislike(x: gr.LikeData):
    indicator, role = "", ""
    if x.liked and not x.index[1]: # user
        indicator = "Good instruction : "
    elif x.liked and x.index[1]:  # AI 
        indicator = "Good response : "
    else:
        indicator = "Bad example: "  
  
    store_conversation(indicator + str(x.value))
    # print(x.index, x.value, x.liked)
    print(indicator + role + str(x.value))


initialize_chat_history_file()


with gr.Blocks() as demo:
    # plot_display = gr.HTML(plot_html)
    chatbot = gr.Chatbot(  # uploaded image of user and cBioportal as avatar 
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=("demo/sample_data/user_avatar.png", 
                    "demo/sample_data/chatbot_avatar.png"),
    )
    chatbot.like(print_like_dislike, None, None)
    gr.ChatInterface(
                    predict, 
                    title="cBioPortal ChatBot",
                    # css="""footer{display:none !important}""",
                    examples=['How to install cBioportal in docker', 'what are important gene list in "Comprehensive molecular portraits of human breast tumors"', 
                    'Give me some studies related to bone', 'can u give me some samples related to TCGA-OR-A5J1-01', 'Tell me a joke', 'how to make a plot of allsample counts in sclc studies'],
                    chatbot=chatbot,
                    )
    
    # write html to add a download button
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
                        <p style="white-space: pre-line;color: #9E9E9E">\nThis chatbot can make mistakes. Check important info.</p>
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
 
demo.launch(allowed_paths=[str(chat_history_file)], share=True)