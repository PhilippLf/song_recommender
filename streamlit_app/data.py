import pandas as pd
import pickle

df_hot = pd.read_csv('../data/hot_songs_spotify.csv')
df_spotify = pd.read_csv('../data/spotify_data_with_clusters.csv')

def load(filename = "filename.pickle"): 
    try: 
        with open(filename, "rb") as f: 
            return pickle.load(f) 

    except FileNotFoundError: 
        print("File not found!")

scaler = load(filename="../Model/scaler.pickle")
kmeans_model = load(filename="../Model/kmeans_7.pickle")