# imports
import os
import openai 
from pdf_handler import pdf_handler
from gpt_handler import gpt_handler

#pedro: Adding this to use azure api instead of openai
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_AZURE_URL") # Azure OpenAI Endpoint
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_AZURE_KEY") # Azure OpenAI Key

# Create an instance of pdf handler
pdf = pdf_handler(pdf_file_path='docs/sphinx_open_DesignersGuide.pdf',
                  embedding_model='gpt-embedding-ada')#this one needs to be the deployed modelo to azuze

embeddings = pdf.get_embeddings()

gpt = gpt_handler(system_role='Use the below information from the "Sphinx Designer`s Guide" to answer the subsequent question. If the answer cannot be found in the information, write "I could not find an answer, please contact GFT."',
                  embeddings=embeddings,
                  gpt_model="gpt-play",
                  embedding_model="gpt-embedding-ada"
                )

while True:
    prompt = input("Make your sphinx question: ")
    if prompt == 'X':
        break
    response=gpt.ask(prompt, print_message=False)
    # Do something with the user input
    print("GPT=>"+response)