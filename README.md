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
