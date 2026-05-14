import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib as job

#Load Dataset
df = pd.read_csv("../dataset/loan_data.csv")

#Features and Target 
x = df.drop('approved', axis = 1)
y = df['approved']

#Split Data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 55)

#Default Model before Tuning
default_model = RandomForestClassifier(random_state = 55)
default_model.fit(x_train,y_train)
default_pred = default_model.predict(x_test)
print("Before Tuning Accuracy:", accuracy_score(y_test, default_pred))

#GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state = 55),
    param_grid,
    cv = 5,
    scoring = 'accuracy',
    verbose = 1
)

grid_search.fit(x_train,y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("\nBest CV Score:", grid_search.best_score_)

#Test Best Model 
best_model = grid_search.best_estimator_
best_pred = best_model.predict(x_test)
print("\nAfter Tuning Accuracy:", accuracy_score(y_test, best_pred))

#Save Model
job.dump(best_model, "../model/loan_model_tuned.pkl")
