import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib as job

#Create Dataset
np.random.seed(42)
n = 1000

data = {
    'age': np.random.randint(22,60,n),
    'salary': np.random.randint(20000,150000,n),
    'credit_score': np.random.randint(300,850,n),
    'loan_amount': np.random.randint(10000,500000,n),
    'years_employed': np.random.randint(0,30,n),
}

df = pd.DataFrame(data)

#Loan approved logic
df['approved'] = (
    (df['salary'] > 50000) &
    (df['credit_score'] > 650) &
    (df['loan_amount'] < df['salary'] * 5) &
    (df['years_employed'] > 2)
).astype(int)

#Save Dataset
df.to_csv("../dataset/loan_data.csv", index = False)

#Features and Target
x = df.drop('approved', axis = 1)
y = df['approved']

#Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Decision Tree
dt_model = DecisionTreeClassifier(random_state = 55)
dt_model.fit(x_train,y_train)
dt_pred = dt_model.predict(x_test)
print("Decision Tree Accuracy:" , accuracy_score(y_test,dt_pred))

#Random Forest
rf_model = RandomForestClassifier(random_state = 55)
rf_model.fit(x_train,y_train)
rf_pred = rf_model.predict(x_test)
print("Random Forest Accuracy:", accuracy_score(y_test,rf_pred))

print("\nDecision Tree Report:")
print(classification_report(y_test,dt_pred))

print("\nRandom Forest Report:")
print(classification_report(y_test,rf_pred))

#Save Random Forest Model
job.dump(rf_model,"../model/loan_model.pkl")