import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter

# ---------------------
# Data Preprocessing
# ---------------------
def cluster_embeddings(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return kmeans.fit_predict(np.stack(embeddings))

def preprocess(data):
    data = data.copy()
    data['Job_Growth_Projection'] = data['Job_Growth_Projection'].astype(str).str.strip().str.title()
    label_map = {'Decline': 0, 'Stable': 1, 'Growth': 2}
    data['Job_Growth_Projection'] = data['Job_Growth_Projection'].map(label_map)
    data.dropna(subset=['Job_Growth_Projection'], inplace=True)
    data.drop(columns=['Location'], inplace=True)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    data['Job_Title_Cluster'] = cluster_embeddings(data['Job_Title'].apply(lambda x: embedder.encode(str(x))), n_clusters=3)
    data['Industry_Cluster'] = cluster_embeddings(data['Industry'].apply(lambda x: embedder.encode(str(x))), n_clusters=3)
    data['Skills_Cluster'] = cluster_embeddings(data['Required_Skills'].apply(lambda x: embedder.encode(str(x))), n_clusters=10)
    data.drop(columns=['Job_Title', 'Industry', 'Required_Skills'], inplace=True)

    data['Company_Size'] = data['Company_Size'].map({'Small': 0, 'Medium': 1, 'Large': 2})
    data['AI_Adoption_Level'] = data['AI_Adoption_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    data['Automation_Risk'] = data['Automation_Risk'].map({'Low': 0, 'Medium': 1, 'High': 2})
    data['Remote_Friendly'] = data['Remote_Friendly'].map({'No': 0, 'Yes': 1})

    # Normalize Salary_USD if it exists
    if 'Salary_USD' in data.columns:
        data['Salary_USD'] = (data['Salary_USD'] - data['Salary_USD'].mean()) / data['Salary_USD'].std()

    data.dropna(inplace=True)

    X = data.drop(columns=['Job_Growth_Projection'])
    y = data['Job_Growth_Projection'].astype(int)
    return X, y

# ---------------------
# Model Definition
# ---------------------
class JobGrowthClassifier(nn.Module):
    def __init__(self, input_dim=8, hidden_dim1=32, hidden_dim2=64, output_dim=3):
        super(JobGrowthClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

# ---------------------
# Training Function
# ---------------------
def train_model(model, train_loader, criterion, optimizer, scheduler, X_val, y_val, patience=10, epochs=100):
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

# ---------------------
# Evaluation Function
# ---------------------
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, predicted, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predicted))

        # Optional: entropy of predictions
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean().item()
        print(f"Mean Prediction Entropy: {entropy:.4f}")

# ---------------------
# Full Pipeline
# ---------------------
if __name__ == '__main__':
    df = pd.read_csv('../data/ai_insights.csv')
    X, y = preprocess(df)
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X.values, y.values, test_size=0.2, stratify=y.values, random_state=42)
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_train_np, y_train_np, test_size=0.2, stratify=y_train_np, random_state=42)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_val = torch.tensor(y_val_np, dtype=torch.long)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train_np)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    model = JobGrowthClassifier(input_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    train_model(model, train_loader, criterion, optimizer, scheduler, X_val, y_val)
    evaluate_model(model, X_test, y_test)