from fastapi import FastAPI
from pydantic import BaseModel
import pickle

from app.padsequence import preder
import numpy as np


# Load the Keras model and tokenizer

with open('app/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

app = FastAPI()

class PredictRequest(BaseModel):
    data: str

@app.get('/')
def root():
    return {'message': 'Toxic comments API'}

@app.post('/predict')
def predict(request: PredictRequest):
    input_text = tokenizer.texts_to_sequences([request.data])
    res = {0:input_text}
    resu = preder(res)
    return {'Toxic':resu[0][0],'Obscene':resu[0][1],'Identity Hate':resu[0][2]}
