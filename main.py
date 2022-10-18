
from fastapi import FastAPI
from requests import request
from simplet5 import SimpleT5
import pandas as pd
app = FastAPI()
generate_mcq_option=SimpleT5()
generate_mcq_option.load_model("t5","simplet5-epoch-1-train-loss-1.1129-val-loss-1.5238")
   

@app.get("/infer")
async def generate_mcqs(text:str):
    #input_data= text.json()
    #input_df = pd.DataFrame([input_data])[0][0]
    mcqs=generate_mcq_option.predict(text)
    return {"mcqs":mcqs}

