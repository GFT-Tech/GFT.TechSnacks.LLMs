# imports
import os
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas   # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

#pedro: Adding this to use azure api instead of openai
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_AZURE_URL") # Azure OpenAI Endpoint
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_AZURE_KEY") # Azure OpenAI Key
embedded_file='parsed_pdf.emb'
if not os.path.exists(embedded_file):
    raise FileNotFoundError("File not found: " + embedded_file) 

df = pandas.read_csv(embedded_file)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# models
EMBEDDING_MODEL = "gpt-embedding-ada"# this is the deployment on Azure OpenAI
GPT_MODEL = "gpt-play"# this is the deployment on Azure OpenAI
# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pandas.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        engine=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pandas.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below information from the "Sphinx Designer`s Guide" to answer the subsequent question. If the answer cannot be found in the information, write "I could not find an answer, please contact GFT."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (
            #gpt-3.5-turbo is fixed here but it is the one used on our azure
            #we use this just to estimate the token size
            num_tokens(message + next_article + question, model="gpt-3.5-turbo")
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pandas.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Sphinx."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

# examples
#strings, relatednesses = strings_ranked_by_relatedness("curling gold medal", df, top_n=5)
#for string, relatedness in zip(strings, relatednesses):
#    print(f"{relatedness=:.3f}")
#    print(string)

while True:
    prompt = input("Make your sphinx question: ")
    if prompt == 'X':
        break
    response=ask(prompt, print_message=False)
    # Do something with the user input
    print("GPT=>"+response)

