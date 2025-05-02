# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from dotenv import load_dotenv
# load_dotenv()

# embedding_model = HuggingFaceEndpointEmbeddings(
#     repo_id="sentence-transformers/all-MiniLM-L6-v2",
#     task="feature-extraction"
# )


# text = "Pakistan in South Asia"
# embedding = embedding_model.embed_query(text)


# doctuments = [
#     "muneeb ur rehman",
#     "stuyding data science",
#     "practical share on github"
# ]




# embedding = embedding_model.embed_documents(doctuments)

# print(str(embedding))
# print(f"Embedding vector size: {len(embedding)}")
# print(len(embedding[0]))





from huggingface_hub import InferenceClient
import os
client = InferenceClient(
    model= "sentence-transformers/all-MiniLM-L6-v2",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)


# print("===========================text_embedding===============================")

# text = "Pakistan in South Asia"

# text_embedd = client.feature_extraction(text)

# print(text_embedd)
# print(type(text_embedd))
# print(len(text_embedd))

print("===========================doc_embedding===============================")


doctuments = [
    "muneeb ur rehman",
    "stuyding data science",
    "graduation from Dawood university"
]

doc_embedding = [ client.feature_extraction(i) for i in doctuments ]

print(doc_embedding)
print(type(doc_embedding))
print(len(doc_embedding))
