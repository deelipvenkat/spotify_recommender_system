# Spotify Music Recommender System
### OBJECTIVE




### ABOUT THE DATASET 


### TECHNOLOGY STACK USED 
```
pip install -r requirements.txt
```

### IMPORTING LIBRARIES


### USING FLASK FOR BUILDING WEB APP

### BUILDING HTML WEB TEMPLATE
Now we build our html webpage template , make sure you add the html doc & it's dependent file in templates folder since most cloud platforms depend on specific file structure for running/deployment of the application.

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

