from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

app = FastAPI()

# Serve /interactives/* (scatter plots .html)
app.mount("/interactives", StaticFiles(directory="interactives"), name="interactives")

@app.get("/", response_class=HTMLResponse)
def home():
    # Serve one of your pages as the default landing
    with open("interactives/index.html", "r", encoding="utf-8") as f:
        return f.read()

class TextIn(BaseModel):
    text: str

class TextOut(BaseModel):
    output: str

@app.post("/predict", response_model=TextOut)
def predict(inp: TextIn):
    # Replace with your real logic
    out = f"You said: {inp.text}"
    return TextOut(output=out)

# /docs will exist automatically
