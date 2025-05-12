import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import collections
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Load the pre-trained model
text_model = SentenceTransformer('all-MiniLM-L6-v2')

data = pd.read_csv('../data/ai_insights.csv')

# Preprocess the Data
def preprocess_data(df):
    label_map = {'Growth': 0, 'Stable': 0, 'Decline': 1}
    df['Job_Growth_Projection'] = df['Job_Growth_Projection'].map(label_map)

    # Map Remote Friendly
    df['Remote_Friendly'] = df['Remote_Friendly'].map({'Yes': 0, 'No': 1})

    # Map categorical levels
    level_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['AI_Adoption_Level'] = df['AI_Adoption_Level'].map(level_map)
    df['Automation_Risk'] = df['Automation_Risk'].map(level_map)

    size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
    df['Company_Size'] = df['Company_Size'].map(size_map)

    # Drop unwanted column
    df = df.drop(columns=['Location'])

    # Combine text columns row-wise
    combined_texts = df['Job_Title'] + " " + df['Industry'] + " " + df['Required_Skills']

    # Generate embeddings
    embeddings = text_model.encode(combined_texts.tolist())

    # Reduce dimensionality of embeddings
    pca = PCA(n_components=60, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)

    # Select structured columns (everything except Job_Title, Industry, Required_Skills, and Label)
    structured_features = df.drop(columns=['Job_Title', 'Industry', 'Required_Skills', 'Job_Growth_Projection']).values

    # Standardize structured features
    scaler = StandardScaler()
    structured_features = scaler.fit_transform(structured_features)

    # Combine with embeddings
    X = np.hstack([structured_features, embeddings_reduced])

    # Labels
    y = df['Job_Growth_Projection'].values

    return X, y

X, y = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
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

# Save Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Reds)
plt.title('Logistic Regression Confusion Matrix')
plt.savefig('../results/model_results/logistic_regression/confusion_matrix.png')

# Save Classification Report As CSV
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('../results/model_results/logistic_regression/classification_report.csv', index=True)

# Save X, Y for future use
np.save('../data/X.npy', X)
np.save('../data/y.npy', y)