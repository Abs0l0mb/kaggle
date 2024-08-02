import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Train the final model using the best parameters
xgb_params = {
    'alpha': 1.302348865795227e-06, 
    'max_depth': 15, 
    'learning_rate': 0.061800451723613786, 
    'subsample': 0.7098803046786328, 
    'colsample_bytree': 0.2590672912533101, 
    'min_child_weight': 10, 
    'gamma': 0.8399887056014855, 
    'reg_alpha': 0.0016943548302122801, 
    'max_bin': 71284,
}

best_xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators = 12000, random_state= 42, eval_metric='auc')
best_xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=500)
y_pred = best_xgb_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy}')

# Load the test data (same as previous code)
test_data = pd.read_csv('test.csv')
test_data = preprocess(test_data)
test_predictions = best_xgb_model.predict(test_data)
submission = pd.DataFrame({
    'PassengerId': pd.read_csv('test.csv')['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('./submission_GBM.csv', index=False)
