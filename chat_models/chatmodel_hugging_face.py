from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
import os
from dotenv  import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(  
repo_id="microsoft/Phi-3-mini-4k-instruct",  
task="text-generation",  
  
)  
chat = ChatHuggingFace(llm=llm)

result = chat.invoke("what is the capital of pakistan")

print(result.content)

