import os
import numpy as np
import pandas as pd
import plotly.express as px
import umap
import faiss
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import joblib
import requests
from bs4 import BeautifulSoup

# Function to fetch suggestions from the internet
def get_web_suggestion(error_message):
    query = f"how to fix {error_message} in programming"
    search_url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        snippets = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
        
        if snippets:
            return snippets[0].text
        else:
            return "No relevant suggestions found online."
    except Exception as e:
        return f"Error fetching suggestions: {str(e)}"

# Load log files
BASE_DIR = "./logs/errorlogs"
log_data = []

for developer in tqdm(os.listdir(BASE_DIR), desc="Reading developer logs"):
    dev_path = os.path.join(BASE_DIR, developer)
    if os.path.isdir(dev_path):
        for file in os.listdir(dev_path):
            if file.endswith(".log"):
                file_path = os.path.join(dev_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    logs = f.readlines()
                    for log in logs:
                        log_data.append({"developer": developer, "log": log.strip(), "timestamp": os.path.getmtime(file_path)})

df = pd.DataFrame(log_data)
if df.empty:
    raise ValueError("No log files found. Please check your directory structure.")

# Convert logs to numeric features
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
X = vectorizer.fit_transform(df["log"])

# Cluster logs using K-Means
num_clusters = min(5, len(df))
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X.toarray())

# Save models
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Reduce dimensions using UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X.toarray())
df["dimension_1"], df["dimension_2"] = X_umap[:, 0], X_umap[:, 1]
joblib.dump(umap_model, "umap_model.pkl")

# Generate suggestions dynamically
model = SentenceTransformer("all-MiniLM-L6-v2")
log_embeddings = model.encode(df["log"].tolist())

dimension = log_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(log_embeddings))

def get_suggestion(log_message):
    query_embedding = model.encode([log_message])
    _, closest_idx = index.search(query_embedding, 1)
    closest_log = df.iloc[closest_idx[0][0]]["log"]
    return get_web_suggestion(closest_log)

df["suggestion"] = df["log"].apply(get_suggestion)

# Create interactive 3D scatter plot
fig = px.scatter_3d(
    df,
    x="developer",
    y="dimension_1",
    z="dimension_2",
    color=df["cluster"].astype(str),
    symbol="developer",
    text=df["timestamp"].astype(str),
    hover_data={"log": True, "suggestion": True},
    title="ُPSP : Defect classification and suggestions",
    labels={"developer": "Developer", "dimension_1": "Feature Dimension 1", "dimension_2": "Feature Dimension 2", "cluster": "Error Cluster"},
)

fig.update_traces(marker=dict(size=8, opacity=0.8), hoverinfo="text")
fig.show()
