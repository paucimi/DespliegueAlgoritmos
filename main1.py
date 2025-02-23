from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline

#Definir pipelines de Hugging Face
sentiment = pipeline("sentiment-analysis")
text_gen = pipeline("text-generation", model="distilgpt2")

# Crear la aplicación FastAPI
app = FastAPI()

class Identity(BaseModel):
    name: str
    surname: Optional[str] = None


@app.get("/hello")
def root():
    return {"message": "¡Hola! Bienvenido a la API de ejemplo."}

@app.post('/test-pydantic')  
def test_pydantic(id: Identity):
    if id.surname is None:
        message = f'Bienvenido a mi API {id.name}'
    else:
        message = f'Bienvenido a mi API {id.name} {id.surname}'
    return {"message": message}

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.get("/penguins")
def read_penguins():
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
    return {"data": df.head(5).to_dict()}

@app.get("/sentiment")
def analyze_sentiment(text: str):
    result = sentiment(text)
    return {"text": text, "analysis": result[0]}

@app.get("/text-generation")
def generate_text(prompt: str, max_length: int = 30):
    result = text_gen(prompt, max_length=max_length, num_return_sequences=1)
    return {"prompt": prompt, "generated_text": result[0]["generated_text"]}

