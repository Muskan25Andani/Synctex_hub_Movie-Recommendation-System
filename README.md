# Synctex_hub_Movie-Recommendation-System

📌 Project Overview

This project implements a content-based movie recommender system using text features and cosine similarity.
It recommends movies based on a combination of overview, genres, keywords, cast, and crew, helping users discover similar movies tailored to their preferences.

📂 Dataset

Source: TMDB 5000 Movies Dataset (CSV files)

tmdb_5000_movies.csv – Movie metadata

tmdb_5000_credits.csv – Movie cast and crew

Features Used:

title – Movie title

overview – Movie description

genres – List of genres

keywords – Movie keywords/tags

cast – Main cast (top 3 actors)

crew – Director name

🎯 Project Objectives

Load and merge the TMDB movies and credits datasets.
Preprocess and clean metadata (handle missing values, parse lists, extract cast/director).
Combine textual features into a single tags column.
Convert text into numerical vectors using TF-IDF.
Compute cosine similarity between movies.
Build a function to recommend top 5 similar movies based on a given title.
Perform EDA: visualize top genres and rating distribution.

⚙️ Project Workflow
1️⃣ Data Loading

Load CSV files using pandas.
Merge datasets on the title column.

2️⃣ Data Preprocessing

Parse genres, keywords, cast, and crew columns using Python ast.literal_eval.
Keep top 3 cast members and director for each movie.
Convert all textual data to lowercase.
Drop movies with missing overviews.

3️⃣ Feature Engineering

Create a tags column combining:
Overview
Genres
Keywords
Cast
Crew

4️⃣ Vectorization

Use TfidfVectorizer with a max of 5000 features and English stop words removed.

Convert tags into numerical vectors.

5️⃣ Similarity Computation

Compute cosine similarity between movie vectors.
Define a recommend(movie_name) function to fetch top 5 similar movies.

6️⃣ Exploratory Data Analysis

Plot Top 10 Movie Genres.
Visualize Movie Rating Distribution.

📊 Libraries Used
pandas
numpy
ast
scikit-learn
matplotlib
seaborn

📁 Project Structure
movie-recommender/
│── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│── src/
│   └── recommender.py
│── requirements.txt

💡 Outcomes

Users can get top 5 recommended movies based on any given movie title.
Movies are compared based on genres, keywords, cast, crew, and overview similarity.
Visualizations provide insights into popular genres and rating distributions.

🚀 Future Enhancements

Build a Flask or Streamlit web app for real-time recommendations.
Use collaborative filtering to improve recommendations.
Add poster images and links in the output.
Integrate IMDb or TMDB APIs for dynamic data updates.

