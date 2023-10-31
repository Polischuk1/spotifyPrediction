# 🎵 Spotify-Based Music Recommendation System 🎵

Welcome to our Spotify-Based Music Recommendation System! 🎉

Are you tired of listening to the same old playlist and looking for fresh, exciting music? Look no further! Our music recommendation system leverages Spotify's vast music database and the power of data analysis to suggest new tracks that match your taste. 

## Table of Contents
1. [**Introduction**](#introduction)
2. [**Getting Started**](#getting-started)
3. [**Data Analysis and Visualization**](#data-analysis-and-visualization)
4. [**Discovering Genres with K-Means**](#discovering-genres-with-k-means)
5. [**Discovering New Music**](#discovering-new-music)
6. [**Example Usage**](#example-usage)
7. [**Contributing**](#contributing)




## 🎵 Introduction

Our music recommendation system is a powerful tool designed to make your music discovery experience more exciting and enjoyable. Here's what you can expect:

- **Data-Driven Recommendations**: We use audio features, genres, and popularity data to recommend songs that match your preferences.
- **Visualize Music Trends**: Explore trends in music popularity, sound features, and genres over time.
- **Discover New Genres**: Our K-Means clustering algorithm groups genres based on their audio features.
- **Personalized Recommendations**: Input your favorite songs, and we'll suggest similar tracks you might love.

## 🚀 Getting Started

1. Make sure you have the required libraries installed. You can install them using `pip` if you haven't already:

   ```bash
   pip install pandas seaborn plotly matplotlib spotipy yellowbrick
   ```

2. Ensure you have the following CSV files in the same directory as your code:
   - `data.csv`: Song data
   - `data_by_artist.csv`: Data by artist
   - `data_by_genres.csv`: Data by genres
   - `data_by_year.csv`: Data by year

3. You need Spotify API credentials to access audio features and other data. Update the `config.py` file with your Spotify API credentials.

4. Run the code provided in your Python environment.

## 📊 Data Analysis and Visualization

Our system starts by analyzing and visualizing various aspects of the music data, such as:

- 📈 Pearson correlation between audio features and popularity.
- 🎵 Popularity trends by decade.
- 📻 Trends in sound features over the years.
- 🏆 Top 10 genres by various features.

## 🌟 Discovering Genres with K-Means

We don't just stop at data analysis; we also discover genres using K-Means clustering:

- 🧬 Preprocesses data by scaling it.
- 💡 Determines the optimal number of clusters using the Elbow Method.
- 🔍 Applies K-Means clustering to genres.
- 🌐 Reduces dimensions using t-SNE for an interactive visualization.

## 🎶 Discovering New Music

The heart of our system is making personalized music recommendations:

- 🕵️‍♀️ Finds audio features for your favorite songs using the Spotify API.
- 📊 Computes the mean vector of your songs' audio features.
- 🎤 Calculates cosine distances between the mean vector and all other songs.
- 🎉 Recommends songs with the closest audio feature profiles.

## 🎧 Example Usage

Here's an example of how to get song recommendations using the system:

```python
recommended_songs = recommend_songs([
    {'name': 'Do You Love Me Still?', 'year': 2006},
    {'name': 'I Drove All Night', 'year': 2006},
    {'name': 'I Wanna Be', 'year': 2007},
    {'name': 'Tonight I Wanna Cry', 'year': 2007},
    {'name': 'My Apocalypse', 'year': 2008},
    {'name': 'Good Fight', 'year': 2014}
], data)

for song in recommended_songs:
    print(f"Recommended Song: {song['name']} ({song['year']})")
```

This will provide a list of recommended songs based on the input songs you provided.


## 🤝 Contributing

We welcome contributions and new ideas to enhance the capabilities of our system. Feel free to fork the repository, make improvements, and submit a pull request.



Enjoy discovering new music with our Spotify-Based Music Recommendation System! 🎶🔥
