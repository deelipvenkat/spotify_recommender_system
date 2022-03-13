# Spotify Music Recommender System
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


### EXPLORATORY DATA ANALYSIS

We have used tableau to perform eda due to it's speed & easy use.
{add tableau images}


### USING K-MEANS CLUSTERING 


### FEATURE SCALING 

We are going to standardize the dataset using standardscaler function available in scikit-learn.

### COMPUTING USER' SAVED TRAKCS MEAN VECTOR BY CLUSTER

In general most of the recommendation engine projects I have seen compute a single mean vector & compute cosine similarity to that mean vector to get recommendation. But Such techniques have some major disadvantages. 

Taking a simple average of all the songs in a user's saved tracks which are of different genres may result in vector which might lead to recommendations very much far off from user's music taste. So to avoid this problem I have used k-means clustering to categorize the songs into groups first ,which is at a decent level maintainig the coherence in individual group & the cosine similarity of the clusters formed from the 170k songs data is far off , indicating disimilarity between clusters formed.

The final model curates playlist from each cluster based on the number of saved songs in a cluster .For example if user 'x' has saved songs in library belonging to cluster A ,B,C in 50%, 30% ,20% respectively , our model will generate recommendations of which 50% are from cluster 'A' , 30% from cluster 'B' and so on.



###  CREATING PICKLE FILES 



### USING FLASK FOR BUILDING WEB APP



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



