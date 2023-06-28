import pinecone
import os
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

_ = load_dotenv(find_dotenv())

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']  # find next to API key in console
)

index_name='gft'

#if you want to cleant he index, uncomment!
index = pinecone.Index(index_name)

namespace=''

query_results = index.query(
    top_k=10000, # 10000 is the max supported! :-(
    vector= [0] * 1536, # embedding dimension
    namespace=namespace,
    include_values=False,
    include_metadata=False)

ids=[]
for item in query_results['matches']:
    ids.append(item['id'])

index.delete(ids, namespace=namespace)
    

