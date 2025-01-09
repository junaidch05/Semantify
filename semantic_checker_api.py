import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from numpy import dot                                           # to calculate the dot product of two vectors
from numpy.linalg import norm
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

model_path  = ('./universal-sentence-encoder_4')

model = tf.saved_model.load(model_path)
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentencePair(BaseModel):
    sentence1: str
    sentence2: str
def embed(input):
  return model(input)

@app.post("/similarity")
def get_similarity(sentence_pair: SentencePair):
    messages=sentence_pair.sentence1,sentence_pair.sentence2
    message_embeddings = embed(messages)
    #converting the sentence pair to vector pair using the embed() function

    a = tf.make_ndarray(tf.make_tensor_proto(message_embeddings))
    #storing the vector in the form of numpy array
    ans = []  
    
    cos_sim = dot(a[0], a[1])/(norm(a[0])*norm(a[1]))
    ans.append(cos_sim)  
    Ans = pd.DataFrame(ans, columns = ['Score'])
    data= Ans['Score'][0]
    data = data+1
    max = data.max().__abs__()
    max = str(max)
    return max