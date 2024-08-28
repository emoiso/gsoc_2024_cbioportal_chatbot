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
import json
import csv


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
    openai_api_type=os.environ["AZURE_OPENAI_API_TYPE"],
    temperature=0  # 0.1
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["DEPLOYMENT_NAME_EMBEDDING"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_4"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

    # pmc: including PMC papers. Choose this if user query mention paper title or anything related to paper
    # documentation: including intro and user guide for cBioportal
    # issue:  including discussions of issues while using cBioportal
classify_chain = (ChatPromptTemplate.from_template(
    """Given the user question and description below, classify it as either being about 
    'pmc', 'documentation', 'issue', 'study' or 'all_chains'. 
    
    Use the chain if user mention it, like use all_chains
    pmc: including PMC papers. Choose this if user query mention paper title or anything related to paper
    documentation: including intro and user guide for cBioportal
    issue:  including discussions of issues while using cBioportal
    all_chains: including all info from other topics. Choose it if u are not sure which topic fits.
    study: get study and sample data by studyID, cancer name, sampleID.
    Question: {question}  
    Chat_history: {chat_history}                  
    Classification:""") | llm_azure | StrOutputParser())


pmc_chain = get_pubmed_chain()
mbox_chain = get_mbox_chain()
documentation_site_chain = get_documentation_site_chain()
all_chain = get_all_chain()
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

    else:
        return all_chain


chat_history = []


def getAnswer(question):
    ans = ({'topic': classify_chain, "question": lambda x: x} | RunnableLambda(route))
    return ans.invoke({"question": question, "chat_history": chat_history})


def load_json_file(path):
    with open(path) as f:
        data = json.load(f)
    return data  


score = 0
data = load_json_file("demo/evaluation.json")
evaluation = {}
source = {}

k = 1
for q in data[:50]:
    options_text = '\n'.join(f"{option}" for i, option in enumerate(q.get('options')))
    question = str(k) + ". " + "Only return the option you think is correct : " + q.get('question') + '\n' + options_text
    evaluation[question] = q.get('correct_answer')
    # source[question] = q.get('data_source')
    k += 1

with open('demo/evaluation_old_mk_k=3_result.csv', 'a', newline='') as csvfile:
    fieldnames = ['Question', 'Your Answer', 'Result']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header row
    writer.writeheader()

    with open('demo/evaluation_old_mk_k=3_result.txt', 'a') as file:
        for question, correct_answer in evaluation.items():
            try:
                ans = getAnswer(question)
                if chain_info == "study" and "I didn't find anything" in ans:
                    second_call = True
                    ans = getAnswer(question)

                response = ans  # + chain_info
            except openai.BadRequestError as e:
                response = "Sorry, no keywords of study found. Please try again with keywords that applies to name and cancer type of the studies" 

            except openai.APIConnectionError as e:
                response = "Server connection error: {}".format(e)

            except Exception as e:
                response = "Error: {}".format(e)
            result = ""
            if response in correct_answer or correct_answer in response:
                score += 1
                result = "Correct"
                file.write(f"Question: {question}\n  Correct_answer: {correct_answer}\n   Your Answer: {response}\nResult: {result}\n\n")
            else:
                result = "Incorrect"
                file.write(f"Question: {question}\n  Correct_answer: {correct_answer}\n   Your Answer: {response}\nResult: {result}\n\n")
            file.write('chain called: {} \n\n'.format(chain_info))
            print(question, response)

            writer.writerow({
                'Question': question + correct_answer,
                'Your Answer': response,
                'Result': result,
                # 'data_source': source[question]
            })

    print("Your total score: ", score)


# def predict(message, history):
#     global chat_history  
#     history_langchain_format = []
#     question = "what is cbioportal"

#     for human, ai in history:
#         history_langchain_format.append(HumanMessage(content=human))
#         history_langchain_format.append(AIMessage(content=ai))

#     history_langchain_format.append(HumanMessage(content=question))
#     try:
#         gpt_response = getAnswer(question)
#         chat_history.extend([HumanMessage(content=question), gpt_response])

#     except openai.BadRequestError as e:
#         gpt_response = "Sorry, no keywords of study found. Please try again with keywords that applies to name and cancer type of the studies" 
   
#     except openai.APIConnectionError as e:
#         gpt_response = "Server connection error: {}".format(e)

#     except Exception as e:
#         gpt_response = "Error: {}".format(e)

#     # chat_history = chat_history[:-1]    
#     partial_message = "Chain : " + chain_info + "\n"

#     for i in range(len(gpt_response)):
#         partial_message += gpt_response[i]
#         time.sleep(0.01)
#         yield partial_message    
#     chat_history = chat_history[-2:] #only keep latest conversation, old messages are removed


# chatbot = gr.Chatbot(  # uploaded image of user and cBioportal as avatar 
#     [],
#     elem_id="chatbot",
#     bubble_full_width=False,
#     avatar_images=("demo/sample_data/user_avatar.png", 
#                    "demo/sample_data/chatbot_avatar.png"),
# )

# gr.ChatInterface(
#                 predict,
#                 title="cBioPortal ChatBot", 
#                 examples=['How to install cBioportal in docker', 'what are important gene list in "Comprehensive molecular portraits of human breast tumors"', 
#                           'Give me some studies related to bone', 'can u give me some samples related to TCGA-OR-A5J1-01', 'Tell me a joke'],
#                 chatbot=chatbot).launch(share=True)
