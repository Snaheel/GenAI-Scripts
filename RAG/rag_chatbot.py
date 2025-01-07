from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)   
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

"""
#First part of adding embeded vectors into postgress DB
#opening the file and splitting text based on \n and creating texts
with open('chatbot_doc.txt', 'r') as f:
    data=f.read()

texts=data.split("\n")

#Creating embeddings
embeddings = OllamaEmbeddings(
    model="llama3.1",
)

vector_list=[]
for i in range(0,len(texts)):
    vec = embeddings.embed_query(texts[i])
    vector_list.append(vec)

# print(len(vector_list[0]))
# print(len(vector_list[-1]))


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

#inseting embeddings for document text
for i in range(0,len(vector_list)):
    querry='INSERT INTO "pgtest".chatbot(texts,embedding) VALUES (\''+texts[i]+'\' , \''+str(vector_list[i])+'\')'
    cursor.execute(querry)
    connection.commit()

cursor.close()
"""

#Second part of fetching relevent info from DB and feeding it to chatbot

#Creating chatbot instance and giving it initial messages
chat = ChatOllama(
    model="llama3.1",
    temperature=0
    )

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi , how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?")
]
#query = "what is a change management process?"
query = "Tell me something about tigers?"


#Fetching Context into from PG vector
dbname = os.getenv('DB_NAME')
user = os.getenv('USER')
password = os.getenv('PASSWORD')
host = os.getenv('HOST')
port= os.getenv('PORT')


connection = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port)

cursor=connection.cursor()

embeddings = OllamaEmbeddings(model="llama3.1")

query_vector = embeddings.embed_query(query)

querry='SELECT * FROM "pgtest".chatbot ORDER BY embedding <-> \''+str(query_vector)+'\' limit 3;'
cursor.execute(querry)
result=cursor.fetchall()

source_knowledge = ""
for i in range(0,len(result)):
    print(result[i][0])
    source_knowledge = source_knowledge+" "+result[i][2]

print(source_knowledge)
cursor.close()

#Giving Extra context to LLM through Human Message
augmented_prompt = f"""Using the contexts below, answer the query. Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn't contain the facts to answer the QUESTION return a generic answer.

Contexts:
{source_knowledge}

Query: {query}"""

prompt = HumanMessage(content=augmented_prompt)
messages.append(prompt)

#Fetching results
res = chat(messages)
print(res.content)
