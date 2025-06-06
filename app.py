from flask import Flask, render_template, jsonify, request
from src.helper import downloadHuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *        # You want to obtain everything (every prompt) inside that file
import os

app = Flask(__name__)

# Attain API Keys from .env
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Download Embeddings from HuggingFace
embeddings = downloadHuggingFaceEmbeddings()

# Initialize Pinecone and Create Index (This is your Knowledge Base)
index_name = "medibot"

# Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# Document Retriever that accesses our Knowledge Base
retriever = docsearch.as_retriever(
    search_type='similarity',
    search_kwargs={'k':3}
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.4,
    max_tokens=500
)

# Set up the System Message
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Initialize RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# First Route: The Consumer will access the interface of the application
@app.route("/")
def index():
    return render_template('chat.html')

# Second Route: This is for the Chat Operation
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    
    return str(response["answer"])


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port= 8080,
        debug= True
    )