import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
from itertools import combinations

def load_and_clean(path):
    df = pd.read_csv(path)
    # Standardize text fields
    df['Genre'] = df['Genre'].str.title().str.strip()
    df['Platform Type'] = df['Platform Type'].str.title().str.strip()
    return df

def preprocess(df):
    numeric_features = [
        'Monthly Listeners (Millions)', 
        'Total Streams (Millions)', 
        'Avg Stream Duration (Min)', 
        'Streams Last 30 Days (Millions)', 
        'Skip Rate (%)'
    ]
    categorical_features = ['Genre', 'Platform Type']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X = preprocessor.fit_transform(df)
    return X, preprocessor

def choose_k(X, k_range=range(2, 11)):
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    # Plot results
    plt.figure()
    plt.plot(list(k_range), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.show()

    plt.figure()
    plt.plot(list(k_range), silhouettes, marker='o')
    plt.title('Silhouette Analysis')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.show()

def cluster_and_pca(df, X, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = km.fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df['PCA1'], df['PCA2'] = coords[:,0], coords[:,1]

    plt.figure()
    for c in range(n_clusters):
        subset = df[df['Cluster'] == c]
        plt.scatter(subset['PCA1'], subset['PCA2'], label=f'Cluster {c}', s=50)
    plt.title('PCA Projection of Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()

    return df

def show_tables(df):
    cluster_counts = df['Cluster'].value_counts().sort_index().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    print("\nCluster Sizes:")
    print(cluster_counts.to_string(index=False))

    genre_dist = df.groupby(['Cluster','Genre']).size().unstack(fill_value=0)
    print("\nGenre Distribution per Cluster:")
    print(genre_dist)

def genre_network(df, threshold=0.7):
    pivot = df.groupby(['Country','Genre'])['Total Streams (Millions)'].sum().unstack(fill_value=0)
    corr = pivot.corr()
    G = nx.Graph()
    for genre in corr.columns:
        G.add_node(genre)
    for g1, g2 in combinations(corr.columns, 2):
        if corr.loc[g1, g2] > threshold:
            G.add_edge(g1, g2, weight=corr.loc[g1, g2])

    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, width=1.2)
    plt.title(f'Genre Correlation Network (r > {threshold})')
    plt.show()

def main():
    path = 'Cleaned_Spotify_2024_Global_Streaming_Data.csv'
    df = load_and_clean(path)
    X, preprocessor = preprocess(df)
    choose_k(X)
    df = cluster_and_pca(df, X, n_clusters=4)
    show_tables(df)
    genre_network(df)

if __name__ == '__main__':
    main()