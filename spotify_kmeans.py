import os
import numpy as np
import pandas as pd 
import seaborn as sns # for making statistical graphics
import plotly.express as px
import matplotlib.pyplot as plt
import spotipy 
import config  # Import the credentials from the separate script


from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict


from yellowbrick.features import parallel_coordinates
from yellowbrick.target import FeatureCorrelation
#from yellowbrick.classifier import classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE #is a tool to visualize high-dimentional data
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data.csv")
data_by_artist = pd.read_csv("data_by_artist.csv")
data_by_genres = pd.read_csv("data_by_genres.csv")
data_by_year = pd.read_csv("data_by_year.csv")


feature_columns = ['valence', 'year', 'acousticness', 'danceability', 'energy', 'key', 'mode', 'tempo', 'explicit']
x = data[feature_columns]
y = data['popularity']
visualizer = FeatureCorrelation(method='pearson')

visualizer.fit(x,y)
visualizer.show()

def music_decade(year):
    period_start = int(year/10)*10
    decade = f"{period_start}s"
    return decade
data['decade'] = data['year'].apply(music_decade)

sns.set(rc={'figure.figsize':(11 ,6)})
sns.countplot(x='decade', data=data)
plt.xticks(rotation=45)  
plt.show()



sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']

# Define your data
x = data_by_year['year']
y = data_by_year[sound_features]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the lines
for feature in sound_features:
    ax.plot(x, data_by_year[feature], label=feature)

# Set labels and legend
ax.set_xlabel('Year')
ax.set_ylabel('Value')
ax.set_title('Sound Features Over the Years')
ax.legend(loc='upper right')

# Show the plot
plt.show()

top10_genres = data_by_genres.nlargest(10, 'popularity')

# Define the data
genres = top10_genres['genres']
valence = top10_genres['valence']
energy = top10_genres['energy']
danceability = top10_genres['danceability']
acousticness = top10_genres['acousticness']

# Set the width of the bars
bar_width = 0.15

# Set the x-axis positions
x = np.arange(len(genres))

# Create subplots
fig, ax = plt.subplots()

# Create bars for each feature
plt.bar(x - 1.5 * bar_width, valence, width=bar_width, label='Valence', align='center')
plt.bar(x - 0.5 * bar_width, energy, width=bar_width, label='Energy', align='center')
plt.bar(x + 0.5 * bar_width, danceability, width=bar_width, label='Danceability', align='center')
plt.bar(x + 1.5 * bar_width, acousticness, width=bar_width, label='Acousticness', align='center')

# Set x-axis labels and tick positions
ax.set_xticks(x)
ax.set_xticklabels(genres, rotation=45, ha="right")
ax.legend(loc='upper left')
# Set y-axis label
plt.ylabel('Value')

# Add a legend
plt.legend(title='Features', loc='upper left')

# Set the title
plt.title('Top 10 Genres by Various Features')

# Show the plot
plt.tight_layout()
plt.show()

#K-means clustering algorithm is used to divide the genres in this dataset into ten clusters 
# based on the numerical audio features of each genres.

cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=6))
])

x = data_by_genres.select_dtypes(np.number)
cluster_pipeline.fit(x)
distortions = []
K_range = range(1,20)
for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    distortions.append(kmeans.inertia_)
    
plt.figure(figsize=(8, 6))
plt.plot(K_range, distortions, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Genres Clustering (Optimal K)')
plt.show()   
#cluster_pipeline.named_steps['kmeans'].fit(x, n_jobs=-1)
data_by_genres['cluster'] = cluster_pipeline.predict(x)

#we use t-sne to reduce it to lower demensions to make it easier to visualize and analyze
tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(x)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = data_by_genres['genres']
projection['cluster'] = data_by_genres['cluster']

# Extract the data for Matplotlib scatter plot
x_values = projection['x']
y_values = projection['y']
cluster_labels = projection['cluster']

# Define a color map for clusters
cluster_colors = ['blue', 'fuchsia', 'skyblue', 'coral', 'yellowgreen', 'crimson', 'chocolate', 'goldenrod', 'salmon', 'teal']

# Create a scatter plot using Matplotlib
plt.figure(figsize=(10, 6))
for cluster_label in cluster_labels.unique():
    cluster_data = projection[projection['cluster'] == cluster_label]
    plt.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster_label}', c=cluster_colors[cluster_label])

plt.xlabel('x')
plt.ylabel('y')
plt.title('T-SNE Visualization of Clusters')
plt.legend(title='Cluster')
plt.show()

#clustering songs with KMeans 

song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=7, verbose=False))
])

x = data.select_dtypes(np.number)
cluster_pipeline.fit(x)
distortions = []
K_range = range(1,22)
for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    distortions.append(kmeans.inertia_)
    
plt.figure(figsize=(8, 6))
plt.plot(K_range, distortions, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for  Clustering (Optimal K)')
plt.show()


x = data.select_dtypes(np.number)
number_columns = list(x.columns)
song_cluster_pipeline.fit(x)
song_cluster_labels = song_cluster_pipeline.predict(x)
data['cluster_label'] = song_cluster_labels

#visualizing the clusters with PCA

x_values = projection['x']
y_values = projection['y']
cluster_labels = projection['cluster']

cluster_colors = ['darkgreen', 'seagreen', 'lightseagreen', 'deepskyblue', 'royalblue', 'navy', 'mediumpurple', 'darkorchid', 'm', 'palevioletred']

plt.figure(figsize=(10,6))
for cluster_label in cluster_labels.unique():
    cluster_data = projection[projection['cluster'] == cluster_label]
    plt.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster_label}', c=cluster_colors[cluster_label])

plt.xlabel('x')
plt.ylabel('y')
plt.title('PCA Visualization of Clusters')
plt.legend(title='Cluster')
plt.show()


sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.environ["SPOTIPY_CLIENT_ID"],
    client_secret=os.environ["SPOTIPY_CLIENT_SECRET"]
))


def find_song(name,year):
    song_data = defaultdict()
    results = sp.search(q='track:{} year: {}'.format(name,year), limit=1)
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


number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=10):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

recommended_songs = recommend_songs([{'name':'Do You Love Me Still?','year': 2006},
                 {'name':'I Drove All Night','year':2006},
                 {'name':'I Wanna Be','year':2007},
                 {'name':'Tonight I Wanna Cry','year':2007},
                 {'name':'My Apocalypse','year':2008},
                 {'name':'Good Fight','year':2014}], data)

for song in recommended_songs:
    print(f"Recommended Song: {song['name']} ({song['year']})")
