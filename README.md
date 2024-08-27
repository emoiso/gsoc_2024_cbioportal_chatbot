# Chatbot for cBioPortal for Cancer Genomics
Contributor: Xinling Wang

Mentors: Augustin Luna, Ruslan Forostianov, Meysam Ghaffari
## About Project :
This project is about build and train a streaming chatbot on four datasets: Documentation site of cBioportal, Google group conversations, PMC papers used in studies, and OPENAPI. Also, the project used route logic to combine 5 chains, and wrote a routing logic function to choose chatbot depending on user query. The chatbot has chat history, indicators, and example questions, download button and footer (disclaimer) in user interface.

### Details about each Chain:
1. **Documentation site** :
   Dataset : https://github.com/cBioPortal/cbioportal/tree/master/docs 
   - This chain can retrieve data from cBioPortal documentation(markdown files) with a URL of document reference. 
   - Markdown files were loaded and splitter by customised Markdown Loader. 
   - Defined a function of adding documentation file url in metadata
   - This database contains 81 markdown files
   - Used Maximal marginal relevance as search type

2.  **Google Conversation chain** :
   Dataset: https://groups.google.com/g/cbioportal 
    - This chain can retrieve data from last 3 years Google Group Conversation from cBioPortal. The format of conversations is Mbox.  
    - Cleaned mbox file by deleting base64 string image and extreme long error messages, which cannot be understand by AI. Convert mbox to json, combined messages from same conversations.
    -  Used a json loader and defined separator, such as new line, to load and split data.
    -  Cleaned the embedding queue in vector database by deleting repeated rows which cause the size to be 100 times larger

3. **PubMed Central papers chain** :
   Dataset:  https://github.com/cannin/gsoc_2024_cbioportal_chatbot/blob/main/demo/pubmed/data/cBioportal_study.json
   - This chain can retrieve data from 200+ PubMed central papers used in 411 cBioPortal studies. 
   - Defined a PubMed Central loader to download pmc papers from S3 and extract full-text to load
   - Used PyMuPDFLoader (can read multi-column) to load pmc papers only in pdf format
   - Added study information as metadata for each chunk in PMC database
   - Contribute the PubMed Central loader to Langchain

4. **cBioportal OPENAPI chain** :
   Dataset:  https://www.cbioportal.org/api/swagger-ui/index.html 
   - This chain can retrieve study and sample data from cBioportal Endpoints 
   - Defined a retriever to handle API call responses
   - Called the LLM a second time to generate a human-readable response.
   - Generated a yaml file to support OpenAPI call
   - Return “Sorry, no keywords of study found. ” when the default of api (return all the studies) is called.
     
5. **Making plots chain** :
   - This chain can retrieve data from defined markdown plotting code example.
   - This chain can generate code of making plot from study and sample informations.
   - Bar chart, pie chart and scatter chart are all available to generate

### Embeddings
Used Chroma to store embeddings. Exported vector databases with metadata in CSV files. https://drive.google.com/drive/folders/12lArAbk3kI8SkPb5W0ynn0eBCjCPWjTV?usp=sharing 
#### Create and persist embedding
```
   embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["DEPLOYMENT_NAME_EMBEDDING"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_4"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
   )


   def create_and_save_embedding(raw_documents, persist_directory, embeddings):
     # This function creates embeddings using Azure OpenAI Embeddings and then saves these embeddings into a vector database using Chroma.
     # Parameters:
     # raw_documents: A list of documents that are split and ready to be processed for embedding.
     # persist_directory: The directory where the vector database will be saved for future use.
     # embeddings: embedding function above

       vectordb = Chroma.from_documents(
           documents=raw_documents,   
           embedding=embeddings,
           persist_directory=persist_directory
       )
       vectordb.persist()
       return vectordb

   def reuse_embedding(persist_directory,embedding_function):
       # This function loads and reuses a previously saved vector database containing document embeddings.
       # Parameters:
       # persist_directory is the path saved in the function above
       # embeddings: embedding function above

       vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
       return vectordb
```


#### Extract embeddings and save in CSV file
```
   #open a Chroma vector database
   db1 = Chroma(persist_directory="your_vectorDB_path")
   count = db1._collection.count()

   def extract_and_write_data(db1, output_file, num_lines=count):
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['id', 'text_model', 'document_content', 'embeddings'])
        
        for i in range(num_lines):
            res = db1._collection.get(limit=1, offset=i, include=["embeddings"])
            doc = db1._collection.get(limit=1, offset=i, include=["documents"])
            metadata = db1._collection.get(limit=1, offset=i, include=["metadatas"])
            text_model = "text-embedding-ada-002"
            for j in range(len(res['ids'])):
                if j < len(res['embeddings']):
                    id_value = res['ids'][j]
                    metadata_content = metadata['metadatas'][j]
                    document_content = doc["documents"][j]  
                    if document_content: 
                        embeddings = ",".join(map(str, res['embeddings'][j]))  # Convert embeddings list to comma-separated string       
                        writer.writerow([id_value, text_model, document_content, embeddings])
   extract_and_write_data(db1, "your_file.csv")
```
## Docker
Dockerized the whole project including all the  embedding databases: 
Docker link: https://hub.docker.com/repository/docker/xinlingwang/chatbot/general

instructions:
1. Recommend create a Azure Openai gpt4 account(chat completions)

2. Create a .env file contains content below:

   AZURE_OPENAI_API_TYPE="azure"
   
   AZURE_OPENAI_ENDPOINT=“YOUR_AZURE_OPENAI_ENDPOINT”
   
   AZURE_OPENAI_API_VERSION_4="YOUR_AZURE_OPENAI_VERSION”
   
   AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_KEY”
   
   DEPLOYMENT_NAME_4="YOUR_AZURE_OPENAI_DEPLOYMENT_NAME"
   
   DEPLOYMENT_NAME_EMBEDDING="YOUR_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
   
   AZURE_OPENAI_API_VERSION_EMBEDDING="YOUR_AZURE_OPENAI_EMBEDDING_VERSION”

3. To Run: docker run -p 7860:7860 --env-file .env xinlingwang/chatbot:with_ls

4. Open browser and go to http://127.0.0.1:7860⁠





https://github.com/user-attachments/assets/39d05d7d-00fb-4267-8714-ca99d266adbb



