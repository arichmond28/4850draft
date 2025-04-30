import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load the dataset
data = pd.read_csv('../data/ai_insights.csv')

# Preprocess the data
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

# Creating Decision Tree Classifier Model
model = DecisionTreeClassifier(random_state=42)

# Fitting The Model To The Training Data
model.fit(X_train, y_train)

# Making Predictions On The Testing Data
y_pred = model.predict(X_test)

# Evaluating The Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Creating Random Forest Classifier Model
rf_model = RandomForestClassifier(random_state=42)

# Fitting The Random Forest Model To The Training Data
rf_model.fit(X_train, y_train)

# Making Predictions On The Testing Data
rf_y_pred = rf_model.predict(X_test)

# Evaluating The Random Forest Model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')
print('Random Forest Classification Report:')
print(classification_report(y_test, rf_y_pred))

# Creating Gradient Boosting Classifier Model
gb_model = GradientBoostingClassifier(random_state=42)

# Fitting The Gradient Boosting Model To The Training Data
gb_model.fit(X_train, y_train)

# Making Predictions On The Testing Data
gb_y_pred = gb_model.predict(X_test)

# Evaluating The Gradient Boosting Model
gb_accuracy = accuracy_score(y_test, gb_y_pred)
print(f'Gradient Boosting Accuracy: {gb_accuracy:.2f}')
print('Gradient Boosting Classification Report:')
print(classification_report(y_test, gb_y_pred))

# Saving GB Classification Report to Text File
with open('../results/gb_classification_report.txt', 'w') as f:
    f.write('Gradient Boosting Classification Report:\n')
    f.write(classification_report(y_test, gb_y_pred))
