from sentence_transformers import SentenceTransformer, util

############################
#Similarity search
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Corpus of documents and their embeddings
corpus = ['Python is an interpreted high-level general-purpose programming language.',
    ' Its design philosophy emphasizes code readability with the use of significant indentation.',
    'The quick brown fox jumps over the lazy dog.']

corpus_embeddings = model.encode(corpus)

queries = ["What is Python?", "What did the fox do?"]
queries_embeddings = model.encode(queries)

hits = util.semantic_search(queries_embeddings, corpus_embeddings, top_k=2)

# Print results of first query
print(f"Query: {queries[0]}")
for hit in hits[0]:
    print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

print(f"Query: {queries[1]}")
for hit in hits[1]:
    print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

######################################
#Image Search
from sentence_transformers import SentenceTransformer, util
from PIL import Image

# Load CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Encode an image
img_emb = model.encode(Image.open('two_dogs_in_snow.png'))

# Encode text descriptions
text_emb = model.encode(['Two dogs in the snow', 'Morging Sunshine', 'A picture of Mumbai at night'])

# Compute cosine similarities 
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)
