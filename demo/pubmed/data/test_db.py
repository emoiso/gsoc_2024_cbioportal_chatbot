# langchain, langchain-community, langchain-chroma
from langchain.vectorstores import Chroma
import chromadb
db1 = Chroma(persist_directory="vectordb/chroma/pubmed/paper_and_pdf")
print(db1._collection.count())

db2 = Chroma(persist_directory="vectordb/chroma/pubmed/pdf")
print(db2._collection.count())


# db2_data=db2._collection.get(include=['documents', 'metadatas', 'embeddings'])
# db1._collection.add(
#      embeddings=db2_data['embeddings'],
#      metadatas=db2_data['metadatas'],
#      documents=db2_data['documents'],
#      ids=db2_data['ids']
# )
# db1._collection.count()

