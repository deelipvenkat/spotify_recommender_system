#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template,redirect
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib
import spotipy
import os
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import startup
import math

app = Flask(__name__)
song_cluster = pickle.load(open('kmeans.pkl', 'rb'))
sc =pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def index():
    response = startup.getUser()
    return redirect(response)
number_cols = ['valence','year','acousticness', 'popularity','danceability', 'duration_ms', 'energy', 'explicit','instrumentalness', 'key', 'liveness', 'loudness', 'mode','speechiness', 'tempo']


def rec_per_cluster(e):
    e=e.dropna(subset=['energy'])
    e['percent']=(e['count']/e['count'].sum())
    e['allowed']=round(e['percent']*50)
    return e

def playlist_cluster_center(play,dj):
# CALCULATING CLUSTER MEANS FOR THE OVERALL SONGS IN THE PLAYLIST.    
    p_mean=pd.DataFrame()
    cluster_count=[]
    a=[_ for _ in range(0,20)]
    a=pd.DataFrame(a)
    inp=get_song_df(play,dj)#program checks for songs in the internal database,if not found then connects to spotify api
    inp_scaled=sc.transform(inp.values)
    inp['cluster']=song_cluster.predict(inp_scaled)
    #song_cluster.predict(sc.transform(get_song_df([{'name':'Blood On The Leaves','year':2013}],data)))
    for i in range(0,20):
        f=inp[number_cols].loc[inp['cluster']==i]
        cluster_count.append(f.shape[0])

        
        p_mean=p_mean.append(pd.DataFrame(np.mean(f.iloc[:,:].values,axis=0).reshape(1,15),columns=number_cols),ignore_index=True)
    cluster_count=pd.DataFrame(cluster_count)    
    p_mean_1=pd.concat([p_mean,a,cluster_count],axis=1)
    p_mean_1.columns=['valence','year','acousticness','popularity', 'danceability', 'duration_ms', 'energy', 'explicit','instrumentalness', 'key', 'liveness', 'loudness', 'mode','speechiness', 'tempo','cluster','count']
    #pd.DataFrame(p_mean_1,columns=['clusters','counts'])
    return p_mean_1


def find_song(name, year):
# finding the audio features of the song using spotify api if the song features are not found in our database.    
    song_data = defaultdict()
    token=startup.getAccessToken()[0]
    sp = spotipy.Spotify(auth=token)
   
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


def get_song_data(song,dj):
    
# searches for the song data in our database using the name & year of song release as unique pair for identification & 
# if not found in our database , it will request the info using the spotify api.
    song_data = dj[(dj['name'].str.lower() == song['name'].lower())& (dj['year'] == song['year'])].iloc[0:1,:]
    if song_data.empty:
        return find_song(song['name'],song['year'])
    else:
        return song_data
        
# creates a dataframe containing all the input songs with their audio_features in a dataframe   
def get_song_df(song_list, dj):

    song_vectors = pd.DataFrame()
    
    for song in song_list:
        song_data = get_song_data(song,dj)
        if song_data is None:
            #print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            
            continue
        song_vector = song_data[number_cols]
        song_vectors=song_vectors.append(song_vector,ignore_index=True)  
    return song_vectors[number_cols]
        
def get_mean_vector(song_list, dj):
    
# computes the raw average of the audio features of all the songs in the playlist.   
    f=get_song_df(song_list,dj)
    g=pd.DataFrame(np.mean(f.iloc[:,:].values,axis=0).reshape(1,15),columns=number_cols)  
    return g


def flatten_dict_list(dict_list):
#[{'a': 1}, {'b': 2}, {'c': 3}] into {'b': 2, 'a': 1, 'c': 3} -each dictionary as seperate list element to a dict

#so for this case , 2 keys will be formed (name,year) & all the values(i.e playlist songs) will be in a list for each keys.

    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, dj, n_songs=20):
# input entry point.(input function)    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list) # flattening the song list into a single  dictionary. 
    
    song_center = get_mean_vector(song_list,dj)# finding mean vector of our input songs.
 
    # standardizing the mean vector & our database songs.
    scaled_data = sc.transform(dj[number_cols])
    scaled_song_center = sc.transform(song_center.values)    
    
    # calculating cosine similarity distance for each song with the mean vector.
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = dj.iloc[index]
    # the below line i think is used for excluding recommendations which were already in the playlist.
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    
    return rec_songs[metadata_cols].reset_index(drop=True) # generating an recommendation output of songs in a dataframe.
    

from sklearn.utils import shuffle
def rec_songs_by_vector(vec,dji,n_songs):
    metadata_cols = ['name', 'year', 'artists','id']
    # standardizing the mean vector & our database songs.
    vc=vec[number_cols]
    scaled_data = sc.transform(dji[number_cols].values)
    scaled_song_center = sc.transform(vc.values)    
    
    # calculating cosine similarity distance for each song with the mean vector.
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = dji.iloc[index]
    
    # the below line i think is used for excluding recommendations which were already in the playlist.
    #rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    
    
    return rec_songs[metadata_cols].reset_index(drop=True) # generating an recommendation output of songs in a dataframe.
    
"""GENERATING A PLAYLIST WITH 50 SONGS AS OUTPUTS TAKING IN MEAN CLUSTERS VECTORS, DATA & ALLOWED AS INPUTS & GIVING THE 
DATAFRAME AS THE OUTPUT. HERE P IS REC_PER_CLUSTER OUTPUT. """    
def spotify_df(p,di):    
    spotify=pd.DataFrame()
    for i in range(0,p.shape[0]):
        h=p.iloc[i:i+1,:]
        c=di.loc[di['cluster']==h['cluster'].values[0]]
        spotify=spotify.append(rec_songs_by_vector(h,c,n_songs=int(h['allowed'].values[0])))
        spotify_rec=shuffle(spotify,random_state=0)
    return spotify_rec.iloc[0:50,:]

@app.route('/callback/')
def callback():
    # Authenticate user with spotify
    startup.getUserToken(request.args['code'])
    tk=startup.getAccessToken()[0]
    sp=spotipy.Spotify(auth=tk)
    """
    save=sp.current_user_saved_tracks()
    saved_count=save['total']
    k=math.ceil(saved_count/20)
    tom=[]
    for i in range(0,k):
        lis=sp.current_user_saved_tracks(offset=(20*i))
        tom.extend(lis['items'])
    
# playlist data in the format used in input function.
    w=[]
    for i in range(0,saved_count):
        w.append({'name':tom[i]['track']['name'],'year':int(tom[i]['track']['album']['release_date'][:4])})
        

    data = pd.read_csv('final_data.csv',nrows=10000)
    
    f=spotify_df(rec_per_cluster(playlist_cluster_center(w,data)),data)    
    f=f.reset_index(drop=True)
    user_id=sp.current_user()['id'] # finding user_id
    playlist_ids=sp.user_playlist_create(user_id,'music_rec',public=True, collaborative=False)['id']# tracking new playlist_id 
    sp.playlist_add_items(playlist_ids,list(f.iloc[:,3].values)) # adding songs in the playlist
    src_html='https://open.spotify.com/embed/playlist/{}?utm_source=generator&theme=0'.format(playlist_ids) """
    return render_template('spotify.html')
    #return render_template('spotify.html',output=src_html)
    #return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)

