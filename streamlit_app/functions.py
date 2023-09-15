import pandas as pd
import numpy as np
import streamlit as st


from sklearn import datasets # sklearn comes with some toy datasets to practice
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import config

import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials

from IPython.display import IFrame



#Initialize SpotiPy with user credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= config.client_id,
                                                           client_secret= config.client_secret))

def get_hot_recommendation(df_hot, track_id):
    """
    This function picks a random song from the Billboard Hot 100 and returns it in a dataframe with one row.
    It does not pick the same song.
    """
    df_without_title = df_hot.drop(df_hot.loc[df_hot['id']==track_id].index)
    df_recommendation = df_without_title.sample()
    
    return df_recommendation


def predicting_cluster_from_song(track_id, kmeans_model, scaler):
    """
    Extracting the audio features for a song from Spotify API given a track ID.
    Predicting the cluster with my trained KMeans model.
    Returning the predicted cluster
    """
    track_df = pd.DataFrame(sp.audio_features(track_id)[0], index=[0])
    
    track_df_num = track_df.drop(['type', 'key', 'mode', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature'], axis = 1)
    
    scaler.fit(track_df_num)
    X_scaled = scaler.transform(track_df_num)
    X_scaled_df = pd.DataFrame(X_scaled, columns = track_df_num.columns)
    
    cluster = kmeans_model.predict(X_scaled_df)
    
    return cluster[0]


def suggested_song_id(df_with_cluster, cluster):
    """
    """
    song_id = df_with_cluster[df_with_cluster['cluster'] == cluster].sample()['id'].iloc[0]
    return song_id
"""
def play_song(track_id):
    return IFrame(src="https://open.spotify.com/embed/track/"+track_id,
       width="320",
       height="80",
       frameborder="0",
       allowtransparency="true",
       allow="encrypted-media",
      )
"""
def play_song(track_id):
    return st.components.v1.iframe(src="https://open.spotify.com/embed/track/"+track_id,
                                   width=320,
                                   height=80,
                                   scrolling=False)