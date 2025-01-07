#This RE has been created using this documentation of PG vector https://github.com/pgvector/pgvector/blob/master/README.md
import os
import httpx
from langchain_ollama import OllamaEmbeddings
import psycopg2
from dotenv import load_dotenv
load_dotenv()

#opening the file and splitting text based on \n and creating documents
with open('re_doc.txt', 'r') as f:
    data=f.read()

texts=data.split("\n")

# print(texts)

#Creating embeddings
embeddings = OllamaEmbeddings(
    model="llama3.1",
)

vector_list=[]
for i in range(0,len(texts)):
    vec = embeddings.embed_query(texts[i])
    vector_list.append(vec)

# print(len(embeddings[1]))

#Enter queries in postgress DB
dbname = os.environ['DB_NAME']
user = os.environ['USER']
password = os.environ['PASSWORD']
host = os.environ['HOST']
port= os.environ['PORT']

connection = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port)

cursor=connection.cursor()
"""
#inseting embeddings for document text
for i in range(0,len(texts)):
    querry='INSERT INTO "pgtest".recomendation(texts,embedding) VALUES (\''+texts[i]+'\' , \''+str(vector_list[i])+'\')'
    cursor.execute(querry)
    connection.commit()
"""
skill="python" #This could also be taken from a command line

q_embed = embeddings.embed_query(skill)

querry='SELECT * FROM "pgtest".recomendation ORDER BY embedding <=> \''+str(q_embed)+'\' LIMIT 1;'
cursor.execute(querry)
result=cursor.fetchall()
res=result[0][2].split(";")
final_res=res[1].split(":")
print("You can make a carrer as an: "+final_res[1])

cursor.close()
