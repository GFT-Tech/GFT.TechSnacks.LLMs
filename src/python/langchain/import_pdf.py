from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
import fnmatch
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.llms import AzureOpenAI
#check .env
_ = load_dotenv(find_dotenv())

# Define the directory to scan
root = 'docs/to_import'
pdfs =[]
# Loop through all subdirectories
for path, subdirs, files in os.walk(root):
    # Loop through all files in the current subdirectory
    for name in files:
        # Check if the file name matches the pattern '*.pdf'
        if fnmatch.fnmatch(name, '*.pdf'):
            # If it does, print the file name
            pdfs.append(os.path.join(path, name))

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_REGION'])

ct=0

# This TokenTextSplitter allowed me some better splitting...
text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100, model_name="text-embedding-ada-002")

embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'],
                              #this makes it work with azure
                              deployment="embedding-tech-snacks",
                              model="text-embedding-ada-002",
                              openai_api_base=os.environ['OPENAI_API_BASE'],
                              openai_api_type=os.environ['OPENAI_API_TYPE']
                              )#1 is azure limitation
for file in pdfs:   
    try:
        print(ct,file,'...')
        namespace = os.path.splitext(os.path.basename(file))[0]
        loader = PyPDFLoader(file)
        pages = loader.load_and_split(text_splitter)       
        print(ct,file,'=>',len(pages),'importing...')
        Pinecone.from_texts([t.page_content for t in pages], embeddings,
                            index_name=os.environ['PINECONE_INDEX'],
                            batch_size=1,
                            namespace=namespace)
    except Exception as e:
        print("An exception occurred:", str(e))
    finally:
        ct+=1



