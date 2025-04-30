# Importing All Required Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading Dataset
data = pd.read_csv('../data/ai_insights.csv')

# Preprocessing Data
def preprocess_data(data):
    # Dropping Unnecessary Columns
    data = data.drop(columns=['Job_Title', 'Industry', 'Location', 'Required_Skills'])

    # Encoding Company Size (Small, Medium Large) as 0, 1, 2
    data['Company_Size'] = data['Company_Size'].map({'Small': 0, 'Medium': 1, 'Large': 2})

    # Encoding AI Adoption Level (Low, Medium, High) as 0, 1, 2
    data['AI_Adoption_Level'] = data['AI_Adoption_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})

    # Encoding Job Growth Projection (Decline, Steady, Growth) as 0, 1, 2
    data['Job_Growth_Projection'] = data['Job_Growth_Projection'].str.strip().str.title()
    data['Job_Growth_Projection'] = data['Job_Growth_Projection'].map({'Decline': 0, 'Stable': 1, 'Growth': 2})

    # Encoding Automation Risk (Low, Medium, High) as 0, 1, 2
    data['Automation_Risk'] = data['Automation_Risk'].map({'Low': 0, 'Medium': 1, 'High': 2})

    # Encoding Remote Friendly (No, Yes) as 0, 1
    data['Remote_Friendly'] = data['Remote_Friendly'].map({'No': 0, 'Yes': 1})

    # Splitting Data Into Features And Target Variable
    X = data.drop(columns=['Job_Growth_Projection'])
    y = data['Job_Growth_Projection']

    return X, y

# Preprocessing Data
X, y = preprocess_data(data)

# Splitting Data Into Training And Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Fitting The Model To The Training Data
model.fit(X_train, y_train)

# Making Predictions On The Testing Data
y_pred = model.predict(X_test)

# Evaluating The Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))