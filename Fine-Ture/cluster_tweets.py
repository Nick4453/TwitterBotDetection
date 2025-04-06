import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Step 1: Load CSV ==========
df = pd.read_csv("preprocessed_all_50_human_tweets.csv")

# Optional: Sample a manageable size
df = df.sample(n=200, random_state=42).copy()

# Clean up weird text encodings
df["Tweet_text"] = df["Tweet_text"].astype(str).str.encode("utf-8").str.decode("utf-8")

# ========== Step 2: Load BERT ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# ========== Step 3: Get Embeddings ==========
def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Generate embeddings (slow!)
df["embedding"] = df["Tweet_text"].apply(get_bert_embedding)

# ========== Step 4: Add Preprocessed Features ==========
feature_cols = [
    "char_count", "word_count", "question_count", "exclamation_count",
    "hashtag_count", "mention_count", "link_count", "polarity", "subjectivity",
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pron_ratio",
    "unique_word_ratio", "stopword_count"
]
numeric_features = df[feature_cols].fillna(0).values

# Standardize numeric features
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_features)

# Stack BERT embeddings + scaled features
X_bert = np.stack(df["embedding"].values)
X_combined = np.hstack((X_bert, scaled_numeric))

# ========== Step 5: KMeans Clustering ==========
kmeans = KMeans(n_clusters=2, random_state=42)
df["cluster"] = kmeans.fit_predict(X_combined)

# ========== Step 6: Visualization ==========
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_combined)
df["pca_x"] = pca_result[:, 0]
df["pca_y"] = pca_result[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="pca_x", y="pca_y", hue="cluster", palette="Set2")
plt.title("Tweet Clusters Based on BERT + Features")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster")
plt.show()

# ========== Step 7: Inspect Tweets ==========
for i in range(2):
    print(f"\n=== Cluster {i} Sample Tweets ===")
    print(df[df["cluster"] == i]["Tweet_text"].head(5).to_string(index=False))
