import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score , f1_score , precision_score

from sklearn.model_selection import RandomizedSearchCV

DATA_PATH = "../Iris.csv"
OUTPUT_MODEL = 'iris.pkl'

models_classification = { 
    'LogicRegression' : (LogisticRegression(), {}),
    'SVC' : (SVC(), {'kernel': ['rbf', 'poly', 'sigmoid']}),
    'DesicionTreeClassifier' : (DecisionTreeClassifier(random_state=42), {'max_depth': [None, 5, 10],'random_state': [42]}),
    'RandomForestClassifier' : (RandomForestClassifier(random_state=42), {'n_estimators': [10, 100],'random_state': [42],'max_depth': [None, 5, 10]}),
    'KNeighborsClassifier' : (KNeighborsClassifier(), {'n_neighbors': np.arange(3, 70, 2),}),
    'GradientBoostingClassifier' : (GradientBoostingClassifier(random_state=42),{'n_estimators': [10, 100],'random_state': [42]}),
    'AdaBoostClassifier': (AdaBoostClassifier(random_state=42), {'n_estimators': [10, 100],'random_state': [42]}),
    }

df = pd.read_csv(DATA_PATH, sep=",")
df = df.drop('Id',axis=1)

train, test = train_test_split(df, test_size = 0.2)

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y=train.Species
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] 
test_y =test.Species  

model_scores = []
for name, (model, params) in models_classification.items():
    pipeline =RandomizedSearchCV(model, params, cv=5)

    pipeline.fit(train_X, train_y)

    ## Prepare parameter
    y_pred = pipeline.predict(test_X)
    accuracy = accuracy_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred,average='macro')
    precision = precision_score(test_y, y_pred,average='macro')
    best_parameter = pipeline.best_params_
    model_scores.append((name,accuracy , f1 , precision,best_parameter))  
 
sorted_models = sorted(model_scores, key=lambda x: x[1])

for model in sorted_models:
    print('Accuracy_score : ', f"{model[0]} is {model[1]: .2f}")
print('\n')

for model in sorted_models:
    print('F1_Score : ', f"{model[0]} is {model[2]: .2f}")
print('\n')

for model in sorted_models:
    print('Precision : ', f"{model[0]} is {model[3]: .2f}")

best_Accuracy_model = max(model_scores, key=lambda x: x[1])

best_f1_model = max(model_scores, key=lambda x: x[2])
 
best_Precision_model = max(model_scores, key=lambda x: x[3])

best_Accuracy_model = max(model_scores, key=lambda x: x[1])
best_model_name = best_Accuracy_model[0]
best_model_instance = models_classification[best_model_name][0]
print('\n')
print("Best Model : ", best_model_name)

## Exporting model
pickle.dump(best_model_instance,open(OUTPUT_MODEL,'wb'))