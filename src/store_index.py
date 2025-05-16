from src.helper import load_pdf_file, downloadHuggingFaceEmbeddings, text_split
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

extracted_data = load_pdf_file(data = 'Data/')
text_chunks = text_split(extracted_data)
embeddings = downloadHuggingFaceEmbeddings()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
print(PINECONE_API_KEY)

pc = Pinecone(api_key= PINECONE_API_KEY)
index_name = "medibot"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

docsearch = PineconeVectorStore.from_documents(
    documents= text_chunks,
    index_name= index_name,
    embedding=embeddings, 
)