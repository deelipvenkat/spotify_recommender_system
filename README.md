# Spotify Music Recommender System (description build in progress)
### OBJECTIVE

We are going to build a content based music recommender system from scratch using spotify api & build a web application through which a user signs-in using his spotify credentials. Inside the application the a curated playlist of 50 songs will be generated based on the user's saved tracks in his spotify library.


### ABOUT THE DATASET 
I have used spotify 170k dataset available in kaggle which contains 170k songs with their features like instrumentalness , artist name , year of release_date etc.

### TECHNOLOGY STACK USED 
I have built this project in jupyter notebook(python3.7). Spotipy library was used for connecting to spotify api & access user data & song features. Oauth library was used for creating a sign in page for our web app using spotify credentials for authorizing client & generating session token. Python flask web framework was used for building the application. The application was deployed in AWS Elastic BeanStalk.

To install all the dependencies for this project, download the requirements.txt file & run the below command line in the terminal.
```
pip install -r requirements.txt
```

### IMPORTING LIBRARIES
```
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
from skimage import io

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import spotipy
import os
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

```

### EXPLORATORY DATA ANALYSIS

We have used tableau to perform eda due to it's speed & easy use.
{add tableau images}



### FEATURE SCALING 

We are going to standardize the dataset using standardscaler function available in scikit-learn.

```
sc=StandardScaler()
sc.fit(data[number_cols].values)
x=sc.transform(data[number_cols].values)

```

### USING K-MEANS CLUSTERING 

```
song_cluster=Pipeline([('kmeans',KMeans(n_clusters=5,init='k-means++',random_state=0))])
song_cluster.fit(x)
data['cluster']=song_cluster.predict(x)
```

## Building song recommendation engine from scratch.

### COMPUTING USER' SAVED TRAKCS MEAN VECTOR BY CLUSTER

In general most of the recommendation engine projects I have seen compute a single mean vector & compute cosine similarity to that mean vector to get recommendation. But Such techniques have some major disadvantages. 

Taking a simple average of all the songs in a user's saved tracks which are of different genres may result in vector which might lead to recommendations very much far off from user's music taste. So to avoid this problem I have used k-means clustering to categorize the songs into groups first ,which is at a decent level maintainig the coherence in individual group & the cosine similarity of the clusters formed from the 170k songs data is far off , indicating disimilarity between clusters formed.

The final model curates playlist from each cluster based on the number of saved songs in a cluster .For example if user 'x' has saved songs in library belonging to cluster A ,B,C in 50%, 30% ,20% respectively , our model will generate recommendations of which 50% are from cluster 'A' , 30% from cluster 'B' and so on.


```
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
```

### collecting songs data 
```
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
```
### Calculating number of recommendations to be generated per cluster.

```
def rec_per_cluster(e):
    e=e.dropna(subset=['energy'])
    e['percent']=(e['count']/e['count'].sum())
    e['allowed']=round(e['percent']*50)
    return e
```
### generating recommendations per cluster

```
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
```

### GENERATING A PLAYLIST WITH 50 SONGS AS OUTPUTS TAKING IN MEAN CLUSTERS VECTORS    
```
def spotify_df(p,di):    
    spotify=pd.DataFrame()
    for i in range(0,p.shape[0]):
        h=p.iloc[i:i+1,:]
        c=di.loc[di['cluster']==h['cluster'].values[0]]
        spotify=spotify.append(rec_songs_by_vector(h,c,n_songs=int(h['allowed'].values[0])))
        spotify_rec=shuffle(spotify,random_state=0)
    return spotify_rec.iloc[0:50,:]
```

###  CREATING PICKLE FILES 


### Authenticating user with spotify login credentials 

```
import base64, json, requests

SPOTIFY_URL_AUTH = 'https://accounts.spotify.com/authorize/?'
SPOTIFY_URL_TOKEN = 'https://accounts.spotify.com/api/token/'
RESPONSE_TYPE = 'code'
HEADER = 'application/x-www-form-urlencoded'
REFRESH_TOKEN = ''

def getAuth(client_id, redirect_uri, scope):
	data = "{}client_id={}&response_type=code&redirect_uri={}&scope={}".format(SPOTIFY_URL_AUTH, client_id, redirect_uri, scope)
	return data

def getToken(code, client_id, client_secret, redirect_uri):
	body = {
		"grant_type": 'authorization_code',
		"code" : code,
		"redirect_uri": redirect_uri,
		"client_id": client_id,
		"client_secret": client_secret
	}


	auth_str = f"{client_id}:{client_secret}"
	encoded = base64.urlsafe_b64encode(auth_str.encode()).decode()

	headers = {"Content-Type" : HEADER, "Authorization" : "Basic {}".format(encoded)}

	post = requests.post(SPOTIFY_URL_TOKEN, params=body, headers=headers)
	return handleToken(json.loads(post.text))

def handleToken(response):
	auth_head = {"Authorization": "Bearer {}".format(response["access_token"])}
	REFRESH_TOKEN = response["refresh_token"]
	return [response["access_token"], auth_head, response["scope"], response["expires_in"]]

def refreshAuth():
	body = {
		"grant_type" : "refresh_token",
		"refresh_token" : REFRESH_TOKEN
	}

	post_refresh = requests.post(SPOTIFY_URL_TOKEN, data=body, headers=HEADER)
	p_back = json.dumps(post_refresh.text)

	return handleToken(p_back)
```

### Generating token to access user data from spotify api
```
from flask_spotify_auth import getAuth, refreshAuth, getToken

#Add your client ID and secret key.

CLIENT_ID = "*********"

CLIENT_SECRET = "*********"

red_uri='http://spotifymusicrecommender-env.eba-mpmh9e3q.ap-south-1.elasticbeanstalk.com/callback/'
#CALLBACK_URL = "https://automated-credit-system.herokuapp.com/"
#Add needed scope from spotify user
SCOPE = "user-library-read playlist-modify-public"
#token_data will hold authentication header with access code, the allowed scopes, and the refresh countdown
TOKEN_DATA = []
"""
def getUser():
	return getAuth(CLIENT_ID, red_uri, SCOPE)

def getUserToken(code):
	global TOKEN_DATA
	TOKEN_DATA = getToken(code, CLIENT_ID, CLIENT_SECRET, red_uri)

"""
def getUser():
	return getAuth(CLIENT_ID, "{}:{}/callback/".format(CALLBACK_URL, PORT), SCOPE)

def getUserToken(code):
	global TOKEN_DATA
	TOKEN_DATA = getToken(code, CLIENT_ID, CLIENT_SECRET, "{}:{}/callback/".format(CALLBACK_URL, PORT))

def refreshToken(time):
	time.sleep(time)
	TOKEN_DATA = refreshAuth()

def getAccessToken():
	return TOKEN_DATA

```




### USING FLASK FOR BUILDING WEB APP

```
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
```
```
application = app = Flask(__name__)
song_cluster = pickle.load(open('kmeans.pkl', 'rb'))
sc =pickle.load(open('scaling.pkl', 'rb'))

@application.route('/')
def index():
    response = startup.getUser()
    return redirect(response)
    
@application.route('/callback/')
def callback():
    # Authenticate user with spotify
    startup.getUserToken(request.args['code'])
    tk=startup.getAccessToken()[0]
    sp=spotipy.Spotify(auth=tk)
    save=sp.current_user_saved_tracks()
    saved_count=save['total']
    k=math.ceil(saved_count/20)
    tom=[]
    for i in range(0,k):
        lis=sp.current_user_saved_tracks(offset=(20*i))
        tom.extend(lis['items'])
    w=[]
    for i in range(0,saved_count):
        w.append({'name':tom[i]['track']['name'],'year':int(tom[i]['track']['album']['release_date'][:4])})
    data = pd.read_csv('final_data.csv')
    f=spotify_df(rec_per_cluster(playlist_cluster_center(w,data)),data)    
    f=f.reset_index(drop=True)
    user_id=sp.current_user()['id'] # finding user_id
    playlist_ids=sp.user_playlist_create(user_id,'music_rec',public=True, collaborative=False)['id']# tracking new playlist_id 
    sp.playlist_add_items(playlist_ids,list(f.iloc[:,3].values)) # adding songs in the playlist
    src_html='https://open.spotify.com/embed/playlist/{}?utm_source=generator&theme=0'.format(playlist_ids)
    return render_template('spotify.html',output=src_html)
if __name__ == "__main__":
    application.run(host='localhost',port=4455,debug=True)
```

### BUILDING HTML WEB TEMPLATE
Now we build our html webpage template , make sure you add the html doc & it's dependent file in templates folder since most cloud platforms depend on specific file structure for running/deployment of the application.

We have used spotify embed to add the ml model curated playlist to our website.
```
<! DOCTYPE  hmtl>
<html>
<head>
<title>Music recommender system</title>
<!--<link rel="shortcut icon" href="./favicon.png" type="image/x-icon">-->
<style>
h1{text-align:center;}
a{text-align:center;}
body{ background-image: url('../templates/back.png');}
</style>
</head>
<br>
<body>
<h1> <a style="text-decoration:none;color:black;font-family:"Garamond" target ="_blank" href="https://github.com/deelipvenkat/spotify_recommender_system">Music Recommender System</a> </h1>
<h2 style="font-family:'Garamond'; align="left">Recommendations For You</h3>
<!--<p><a style="color:blue;" target="_blank" href="https://www.linkedin.com/in/deelip-venkat"> About Me </a></p> -->
<iframe style="border-radius:12px" src={{output}} width="100%" height="100%" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>
</body>
</html>
  ```
Also we create a folder named 'static' in our root directory which contains the background image for our web app.


### CREATING .ebextension DIRECTORY 

We can configure our Elastic Beanstalk environment & customize aws resources by adding .config files in .ebextensions directory.

LET'S configure our python environment which is going to run the application in aws by creating a python.config & adding the below code.
```
option_settings:
  "aws:elasticbeanstalk:container:python":
    WSGIPath: application.py
```   
   
### INCREASING TIMEOUT FOR THE APPLICATION 

After the user signs into the application , the user's saved tracks from the library is collected through the api & is feed into the k means clustering model to categorize the songs into groups & cosine similarity of saved tracks with the initial dataset is computed to find similar songs to the liking of the user. All of these computations take about 2 minutes. Most of the cloud platforms keep the default time-out to 50 seconds to avoid any malicious code stuck in loop & costing more cloud expenses. So in our case we are going to increase our timeout to suit our needs.

So let's create another config file which tells our aws to increase timeout limit. Addd the  In this case we have increased it to 300 seconds.


```
option_settings:
  aws:elasticbeanstalk:command:
    Timeout: 300
```

### 

```
web: uwsgi --http :8000 --wsgi-file application.py
```

### 


### REFERENCES



