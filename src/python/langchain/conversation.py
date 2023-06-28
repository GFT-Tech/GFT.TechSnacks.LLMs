import os
from langchain import PromptTemplate
import openai
import pinecone
from fastapi.middleware.cors import CORSMiddleware
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT
from dotenv import load_dotenv, find_dotenv
from langchain.llms import AzureOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA



#check .env
_ = load_dotenv(find_dotenv())

# Open AI Chat Completion LLM Model from Langchain, with Streaming enabled
openai_llm = AzureOpenAI(
    deployment_name="gpt-tech-snacks",
    model_name="text-embedding-ada-002",
    temperature=0
)

embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'],
                              #this makes it work with azure
                              deployment="embedding-tech-snacks",
                              model="text-embedding-ada-002",
                              openai_api_base=os.environ['OPENAI_API_BASE'],
                              openai_api_type=os.environ['OPENAI_API_TYPE']
                              )#1 is azure limitation
TEXT_FIELD = "text"
index = pinecone.Index(os.environ['PINECONE_INDEX'])
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

vectorstore = Pinecone(
    index, embeddings.embed_query, TEXT_FIELD,
    namespace="sphinx"
)

qa = RetrievalQA.from_chain_type(
    llm=openai_llm,
    chain_type="stuff",    
    retriever=vectorstore.as_retriever(),
    memory=memory
)
while True:
    query = input("> ")
    response = qa.run(query)
    print(response)
