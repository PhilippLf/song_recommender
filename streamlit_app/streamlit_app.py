import streamlit as st
import functions
import data
import config
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#Initialize SpotiPy with user credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= config.client_id,
                                                           client_secret= config.client_secret))

st.title('Song Recommender')
st.subheader('Based on the Spotify API and KMeans Clustering')

counter=0
#while True:
title = st.text_input('Enter a song you like', key=counter)
if title:
    results = sp.search(q=f"{title}",limit=5, market='DE')
    song_name = results["tracks"]["items"][0]["name"]
    artist_name = results["tracks"]["items"][0]["artists"][0]["name"]
    track_id = results["tracks"]["items"][0]["id"]
    st.text(f"Do you mean {song_name} from {artist_name}?")
    functions.play_song(track_id)
    yes = st.button('Yes')
    no = st.button('No')
    if yes:
        if track_id in list(data.df_hot['id']):
            df_recommendation = functions.get_hot_recommendation(data.df_hot, track_id)
            st.text(f"You might like {df_recommendation['track_name'].iloc[0]} from {df_recommendation['artist'].iloc[0]}!")
            functions.play_song(df_recommendation['id'].iloc[0])
        else:
            cluster = functions.predicting_cluster_from_song(track_id, data.kmeans_model, data.scaler)
            song_id = functions.suggested_song_id(data.df_spotify, cluster)
            st.text("You might like this song:")
            functions.play_song(song_id)
            #break
    if no:
        st.text("Please specify your search request or choose another song.")
        counter+=1