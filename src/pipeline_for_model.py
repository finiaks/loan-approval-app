import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
import joblib as job

#Load Dataset
df = pd.read_csv("../dataset/loan_data.csv")

#Features and Target
x =df.drop('approved', axis = 1)
y = df['approved']

#Split Data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 55)

#Create Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators = 200,
        max_depth = 10,
        min_samples_split = 5,
        random_state = 55
    ))
])

#Train Pipeline
pipeline.fit(x_train,y_train)

#Predict
y_pred = pipeline.predict(x_test)

#Results
print("Pipeline Accuracy:", accuracy_score(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#Save Pipeline
job.dump(pipeline, "../model/loan_model_pipeline.pkl")
