# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# load_dotenv()

# embedd_model = HuggingFaceEndpointEmbeddings(
#     repo_id="sentence-transformers/all-MiniLM-L6-v2",
#     task="feature-extraction",
    
# )

# sentences = [
#     "Pakistan's capital city is Islamabad.",
#     "The Indus River flows through Pakistan.",
#     "Pakistan shares a border with India.",
#     "The national language of Pakistan is Urdu.",
#     "Pakistan was founded in 1947."
# ]

# query = "When was Pakistan founded?"


# doc_embeddings = embedd_model.embed_documents(sentences)

# query_embedding = embedd_model.embed_query(query)
# print(cosine_similarity([query_embedding], doc_embeddings))


from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# from dotenv import load_dotenv
import os

# load_dotenv()

client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

sentences = [
    "Pakistan's capital city is Islamabad.",
    "The Indus River flows through Pakistan.",
    "Pakistan shares a border with India.",
    "The national language of Pakistan is Urdu.",
    "Pakistan was founded in 1947."
]

query = "When was Pakistan founded?"

# Embed documents
doc_embeddings = [client.feature_extraction(sentence) for sentence in sentences]

# Embed query
query_embedding = client.feature_extraction(query)

# Compute similarity
similarities = cosine_similarity([query_embedding], doc_embeddings)
print(similarities)
print(similarities[0])
print(list(enumerate(similarities[0])))# putting in enumerate function 

print(sorted(list(enumerate(similarities[0])) , key=lambda x : x[1]))# sorted on the base of second arg (0, np.float32(0.57881653))

print(sorted(list(enumerate(similarities[0])) , key=lambda x : x[1]))
print(sorted(list(enumerate(similarities[0])) , key=lambda x : x[1])[-1]) # now getting the last value that is maxvalue


i , socore = sorted(list(enumerate(similarities[0])) , key=lambda x : x[1])[-1]

print(query)
print(sentences[i])
print(socore)


