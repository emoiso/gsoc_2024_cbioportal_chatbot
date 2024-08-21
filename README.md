# Chatbot for cBioPortal for Cancer Genomics
Contributor: Xinling Wang

Mentors: Augustin Luna, Ruslan Forostianov, Meysam Ghaffari
## About Project :
This project has four chatbots which was trained on different datasets. Also, the project used route logic to combine 4 chatbots, and wrote a defined function to choose chatbot depending on user query.

Each chatbot has its own loader, metadate, splitter, prompt, retriever and independent vectorDB.

### Details about each chatbot:
1. Dataset : Documentation site: https://github.com/cBioPortal/cbioportal/tree/master/docs 
   
2. Dataset : Last 3 years of Google Conversation : https://groups.google.com/g/cbioportal 

3. Dataset : 261 PubMed Central papers from all (~400) studies : https://github.com/cannin/gsoc_2024_cbioportal_chatbot/blob/main/demo/pubmed/data/cBioportal_study.json

4. Dataset : cBioportal Endpoints (getAllstudiesUsingGET and getSamplesBykeywordsUsingGET)  https://www.cbioportal.org/api/swagger-ui/index.html 

### Embeddings
Used Chroma to store embeddings. Exported vector databases with metadata in CSV files. https://drive.google.com/drive/folders/12lArAbk3kI8SkPb5W0ynn0eBCjCPWjTV?usp=sharing 
#### Create and reuse embedding
```
   embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["DEPLOYMENT_NAME_EMBEDDING"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_4"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
   )

  # call this only for one time to create embeddings
   def create_and_save_embedding(raw_documents, persist_directory):
       vectordb = Chroma.from_documents(
           documents=raw_documents,   # raw_documents is splitted and loaded documents
           embedding=embeddings,
           persist_directory=persist_directory
       )
       vectordb.persist()
       return vectordb

   # call this to reuse vectordb 
   def reuse_embedding(persist_directory,embedding_function):
       vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
       # persist_directory is the path saved in the function above
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
            # metadata = db1._collection.get(limit=1, offset=i, include=["metadatas"])
            text_model = "text-embedding-ada-002"
            for j in range(len(res['ids'])):
                if j < len(res['embeddings']):
                    id_value = res['ids'][j]
                    # metadata_content = metadata['metadatas'][j]
                    document_content = doc["documents"][j]  
                    if document_content: 
                        embeddings = ",".join(map(str, res['embeddings'][j]))  # Convert embeddings list to comma-separated string       
                        writer.writerow([id_value, text_model, document_content, embeddings])
   extract_and_write_data(db1, "mbox.csv")
```
