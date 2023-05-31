from fastapi import FastAPI, Response
import os
import openai
import os
import json
from fastapi.middleware.cors import CORSMiddleware

#pedro: Adding this to use azure api instead of openai
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_AZURE_URL") # Azure OpenAI Endpoint
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_AZURE_KEY") # Azure OpenAI Key

PRESENTER_SYSTEM_ROLE ="""
You are an friendly assistant that produces Reveal.Js Slides and follow ALL these guidelines:
- ONLY and EXCLUSIVELY use Reveal.js's Markdown format, without ANY Extra text
- Use Bullet Points and keywords, no more than 10 words per bullet
- Use ONLY ## as title on each slide
- Never use # or ### as title of each slide
- Never generate an Introduction Slide
- Never generate an Agenda Slide 
- Never generate an Title Slide 
- Add --- to split slides 
- On Every slide, add a line before the '# title' with '<!-- .element: data-background-image="IMAGE_HERE" -->'. Replace "IMAGE_HERE" with images from the List under "Images:" 
Images: 
- https://www.gft.com/.imaging/focalpoint/1600x960/dam/jcr:6400a54c-0eae-44bf-a692-838d1ea49138/gft-group_financial-figures-2021_record-year_light.webp
- https://www.gft.com/.imaging/focalpoint/1600x960/dam/jcr:487e72b0-fc4c-4853-9484-1e6604aa45a4/220317_GFT-Shaping-the-future_Header.webp
- https://www.gft.com/.imaging/focalpoint/1600x960/dam/jcr:8f71f9f4-fbbf-4bd1-a42f-7fa9b81ed026/gft-key-visual-contact.webp
- https://www.gft.com/.imaging/focalpoint/1600x960/dam/jcr:435eabfb-88a1-4fa0-94cb-348da6ebc393/gft-blockchain-02.webp
- https://www.gft.com/.imaging/focalpoint/1600x960/dam/jcr:2c1b77b5-ac18-43c8-a99e-b6c914bbaeea/GettyImages-979512232_retouch.webp
- https://www.gft.com/.imaging/focalpoint/1600x960/dam/jcr:e5a10797-6148-48a8-af0e-5ef35c1a17d5/gft-key-visual-sphinx-open.webp
- https://www.gft.com/.imaging/focalpoint/1600x960/dam/jcr:ba906e9e-aa6c-481e-be18-46d29a0830c0/Stage-Header_Image_greencoding-1.webp
- https://www.gft.com/.imaging/focalpoint/1600x960/dam/jcr:b5b283c5-208c-4e2b-a079-1d465b5e7fa8/shutterstock_715237756.webp
- https://www.gft.com/.imaging/focalpoint/1600x960/dam/jcr:9f3198d7-a31d-41cc-8809-90836376ff6b/gft-key-visual-alper-seguros-embraces-modern-technologies-to-position-for-growth.webp 
"""

app = FastAPI()
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response
# pcin-net.local
origins = [
    "*"
]
#
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/presentation')
def presentation():    

    user_context = [
        {"role":"system","content":PRESENTER_SYSTEM_ROLE},
        {"role":"user","content":"5 Markdown(Reveal.JS) Slides explaining the top 5 main topics to explan how Embeddings works on LLMs"}]
    response = openai.ChatCompletion.create(
        engine="gpt-play",
        messages=user_context,
        temperature=0
    )
    response_json_string = response["choices"][0]["message"]["content"]
    return Response(content=response_json_string, media_type="text/template")
    
@app.post('/presentation-create')
def process_message(payload: dict):    
    user_context = payload.get('messages', [])    
    user_message_content = user_context[-1]["content"]    
    user_context.insert(0, {"role": "system", "content": PRESENTER_SYSTEM_ROLE })
    response = openai.ChatCompletion.create(
        engine="gpt-play",
        messages=user_context,
        temperature=0
    )
    response_json_string = response["choices"][0]["message"]["content"]

    return Response(content=response_json_string, media_type="text/template")

@app.post('/ask')
def process_message(payload: dict):    
    user_context = payload.get('messages', [])    
    user_message_content = user_context[-1]["content"]    
    user_context.insert(0, {"role": "system", "content": PRESENTER_SYSTEM_ROLE })
    response = openai.ChatCompletion.create(
        engine="gpt-play",
        messages=user_context,
        temperature=0
    )
    response_json_string = response["choices"][0]["message"]["content"]

    return Response(content=response_json_string, media_type="text/template")

