from transformers import pipeline
import torch

##############################################
#using pipeline which is a wrapper around all transformers
classifier=pipeline("sentiment-analysis")
result=classifier("Wow what a day!")
print(result)

#Example of a text generation model
classifier=pipeline("text-generation",model="distilgpt2") 
result=classifier("Today is a beatiful shining day", max_length=20,num_return_sequences=2)
print(result)

#zero-shot classification is about knowing which type of text is provided and then we get different scores for each label
classifier=pipeline("zero-shot-classification") 
result=classifier("Today is a beatiful shining day", candidate_labels=["education","bussiness","sports","weather"])
print(result)

##############################################
# Generating a summary without creating embeddings from model
model_name = "google/pegasus-xsum"
model = pipeline("summarization", model=model_name)

input_text = "The sun was shining brightly in the clear blue sky, casting a warm glow over the lush green grass. A gentle breeze rustled the leaves of the nearby trees, carrying the sweet scent of blooming flowers through the air. The temperature was just right, not too hot and not too cold, perfect for a leisurely stroll or a picnic in the park. The sky was a brilliant blue, with only a few wispy clouds drifting lazily across it, adding a touch of serenity to the scene. It was a beautiful day, one of those days where everything seemed right with the world, and all that was needed was a comfortable spot to relax and soak up the sunshine."

summary = model(input_text, max_length=100, num_beams=4, early_stopping=True)

# print(summary)
print(summary[0]['summary_text'])

##############################################
#Creating a a document summarizer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

input_text = "London University, formally known as the University of London, is a public research university located in the heart of London, England. Founded in 1836, it is one of the oldest and largest universities in the UK, with a rich history of academic excellence and innovation. With a diverse range of undergraduate and graduate programs across various disciplines, including arts, humanities, sciences, engineering, and law, London University is a popular choice for students from around the world. The university is composed of several colleges and institutes, including University College London, King's College London, and Royal Holloway, among others, each with its own unique character and strengths. London University is renowned for its research quality, with many of its departments ranked among the world's top 10 in their respective fields. Its proximity to the British Museum, the British Library, and other cultural institutions also provides students with unparalleled opportunities for research and study."

model_name = "google/pegasus-xsum"

tokenizer = PegasusTokenizer.from_pretrained(model_name) #loading tokenizer
model = PegasusForConditionalGeneration.from_pretrained(model_name) #loading Model

embeddings = tokenizer.encode(input_text, return_tensors="pt", truncation=True) #To create embeddings
summary_ids = model.generate(embeddings, num_beams=4, max_length=100) # The embeddings are given to model for summarizing the text

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True) #Summary tokens are decoded

print(summary)

##############################################
#Generating using ollama locally
#Step 1) dowload ollama and pull lamma3.1 model
#Step 2) Run the below code
from langchain_ollama import OllamaEmbeddings

embeddings = embeddings = OllamaEmbeddings(model="llama3.1")

text="Hello World"
single_vector = embeddings.embed_query(text)

print(len(single_vector))
