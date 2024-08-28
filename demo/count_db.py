from langchain.vectorstores import Chroma
import chromadb
import csv


db1 = Chroma(persist_directory="your_vectorDB1_path")
print(db1._collection.count())

count = db1._collection.count()

db2 = Chroma(persist_directory="your_vectorDB2_path")
print(db2._collection.count())

# add vectorDB2 into vectorDB1
# db2_data=db2._collection.get(include=['documents', 'metadatas', 'embeddings'])
# db1._collection.add(
#      embeddings=db2_data['embeddings'],
#      metadatas=db2_data['metadatas'],
#      documents=db2_data['documents'],
#      ids=db2_data['ids']
# )
# print(db1._collection.count())
