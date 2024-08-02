import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Preprocess the data : fill missing values, categorize continuous variables, map non-numeric values, drop irrelevant columns
def preprocess(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Fare'] = pd.cut(data['Fare'], bins=[-float('inf'), 7.91, 14.454, 31, float('inf')], labels=[0, 1, 2, 3]).astype(int)
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Age'] = pd.cut(data['Age'], bins=[-float('inf'), 16, 32, 48, 64, float('inf')], labels=[0, 1, 2, 3, 4]).astype(int)
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data.drop(columns=['Age', 'SibSp', 'PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    return data

train_data = pd.read_csv('train.csv')
train_data = preprocess(train_data)
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier()

# Use grid search to find the best number of estimators
param_grid = {
    'n_estimators': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Accuracy: {best_score}')

# Use a random forest with the best number of estimators
best_rf = RandomForestClassifier(n_estimators=best_params['n_estimators'])
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy}')

# Make submission
test_data = pd.read_csv('test.csv')
test_data = preprocess(test_data)
test_predictions = best_rf.predict(test_data)
submission = pd.DataFrame({
    'PassengerId': pd.read_csv('test.csv')['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('./submission_RF.csv', index=False)
print('Submission file created: submission_RF.csv')