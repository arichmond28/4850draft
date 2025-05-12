import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Load Data
data = pd.read_csv('../data/ai_insights.csv')

# Load the pre-trained model
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocess the Data
def preprocess_data_for_trees(df, n_components=60):
    label_map = {'Growth': 0, 'Stable': 0, 'Decline': 1}
    df['Job_Growth_Projection'] = df['Job_Growth_Projection'].map(label_map)
    df['Remote_Friendly'] = df['Remote_Friendly'].map({'Yes': 0, 'No': 1})
    level_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['AI_Adoption_Level'] = df['AI_Adoption_Level'].map(level_map)
    df['Automation_Risk'] = df['Automation_Risk'].map(level_map)
    size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
    df['Company_Size'] = df['Company_Size'].map(size_map)
    df = df.drop(columns=['Location'])

    combined_texts = df['Job_Title'] + " " + df['Industry'] + " " + df['Required_Skills']
    embeddings = text_model.encode(combined_texts.tolist())

    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)

    structured_features = df.drop(columns=['Job_Title', 'Industry', 'Required_Skills', 'Job_Growth_Projection']).values

    X = np.hstack([structured_features, embeddings_reduced])
    y = df['Job_Growth_Projection'].values

    return X, y

# Preprocess the data
X, y = preprocess_data_for_trees(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=15, min_samples_split=2)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
# Create and train the model
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=50, max_depth=10, min_samples_split=2, min_samples_leaf=2)
rf_model.fit(X_train, y_train)
# Make predictions
y_pred_rf = rf_model.predict(X_test)
# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
print('Random Forest Classification Report:')
print(classification_report(y_test, y_pred_rf))
print('Random Forest Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_rf))

# XGBoost Classifier
from xgboost import XGBClassifier
# Create and train the model
xgb_model = XGBClassifier(random_state=42, scale_pos_weight=1, max_depth=10, min_child_weight=2)
xgb_model.fit(X_train, y_train)
# Make predictions
y_pred_xgb = xgb_model.predict(X_test)
# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'XGBoost Accuracy: {accuracy_xgb:.2f}')
print('XGBoost Classification Report:')
print(classification_report(y_test, y_pred_xgb))
print('XGBoost Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_xgb))

# Save Confusion Matrix
cm = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Reds)
plt.title('XGBoost Confusion Matrix')
plt.savefig('../results/model_results/decision_trees/confusion_matrix.png')

# Save Classification Report As CSV
report = classification_report(y_test, y_pred_xgb, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('../results/model_results/decision_trees/classification_report.csv', index=True)