import pandas as pd
import pickle 
import uvicorn

from pydantic import BaseModel
from sklearn.model_selection import train_test_split 
from fastapi import FastAPI

app = FastAPI()

MODEL_PATH = "../modelling/iris.pkl"
DATA_PATH = "../Iris.csv"

df = pd.read_csv(DATA_PATH, sep=",")
df = df.drop('Id',axis=1)

train, test = train_test_split(df, test_size = 0.3)

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y = train.Species

irish_type = {1 : "Iris-setosa", 
            2 : "Iris-versicolor", 
            3 : "Iris-virginica"}

# Load and Fit Model
model = pickle.load(open(MODEL_PATH,'rb'))
model.fit(train_X, train_y)

class predict(BaseModel):
    data: list = None

@app.post("/predict")
async def predit(input : predict):
    if input.data is None : 
        return None

    ## Formating input
    all_data = []
    for i in input.data: 
        data = []
        for k,v in i.items():
            data.append(v)
        all_data.append(data)

    df = pd.DataFrame(all_data, columns=["SepalLengthCm",  "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

    result = model.predict(df)

    final_result = []
    for k, v in irish_type.items() :
        for i in result : 
            if i == v : 
                final_result.append(k)

    return final_result
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)