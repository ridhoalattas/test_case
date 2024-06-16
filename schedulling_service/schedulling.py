import time
import schedule
import pytz
import pandas as pd
import pickle 

from sklearn.model_selection import train_test_split 
from datetime import datetime
import database as mongo

MODEL_PATH = "../modelling/iris.pkl"
DATA_PATH = "../Iris.csv"
TIME_TO_EXECUTE = "02:09"

irish_type = {1 : "Iris-setosa", 
            2 : "Iris-versicolor", 
            3 : "Iris-virginica"}

inputs = [{
    "id" : "asdf",
    "sepal_length" : 6.0, 
    "sepal_width" : 3.0, 
    "petal_length" : 4.0, 
    "petal_width" : 1.5
},{
    "id" : "1234",
    "sepal_length" : 6.1, 
    "sepal_width" : 3.2, 
    "petal_length" : 4.3, 
    "petal_width" : 1.1
}]

df = pd.read_csv(DATA_PATH, sep=",")
df = df.drop('Id',axis=1)

train, test = train_test_split(df, test_size = 0.3)

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y = train.Species

# Load and Fit Model
model = pickle.load(open(MODEL_PATH,'rb'))
model.fit(train_X, train_y)

def job():
    ## Formating input
    all_data = []
    id = []
    for i in inputs: 
        data = []
        for k,v in i.items():
            if k == "id" : 
                if v not in id : 
                    id.append(v)
            data.append(v)
        all_data.append(data)
        
    df = pd.DataFrame(all_data, columns=["id", "SepalLengthCm",  "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

    df_to_predict = df.drop("id", axis=1)
    result = model.predict(df_to_predict)

    final_result = []
    for k, v in irish_type.items() :
        for i in result : 
            if i == v : 
                final_result.append(k)
                
    df_result = pd.DataFrame(columns=["executed_at", "id", "class"])

    time_format = '%Y-%m-%d %H:%M:%S %Z%z'
    time = datetime.now(pytz.timezone('Asia/Jakarta'))

    for idx, (id, predicted) in enumerate(zip(id, final_result)) :
        row = [time.strftime(time_format), id, predicted]
        
        index_loc = df_result.index.max()

        if pd.isna(index_loc):
            df_result.loc[0] = row
        else:
            df_result.loc[index_loc + 1] = row

    ## Insert to database
    results = {}
    for k, v in df_result.T.to_dict().items():
        results[str(k)] = k, v
    mongo.insert_data(results)
    print(df_result)
    return True

schedule.every(10).minutes.do(job)
schedule.every().hour.do(job)
schedule.every().day.at(TIME_TO_EXECUTE).do(job)

while 1:
    schedule.run_pending()
    time.sleep(1)