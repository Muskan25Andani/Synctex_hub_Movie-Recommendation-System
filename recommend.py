import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# LOAD DATA
# =========================
movies = pd.read_csv("archive/tmdb_5000_movies.csv")
credits = pd.read_csv("archive/tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")


# =========================
# CLEANING FUNCTIONS
# =========================
def convert(text):
    return [i["name"] for i in ast.literal_eval(text)]

def get_top_cast(text):
    cast = []
    for i in ast.literal_eval(text):
        cast.append(i["name"])
        if len(cast) == 3:
            break
    return cast

def get_director(text):
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            return i["name"]
    return ""


# =========================
# METADATA CLEANING
# =========================
movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(get_top_cast)
movies["crew"] = movies["crew"].apply(get_director)

movies = movies.dropna(subset=["overview"])


# =========================
# FEATURE ENGINEERING
# =========================
movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"].apply(lambda x: " ".join(x)) + " " +
    movies["keywords"].apply(lambda x: " ".join(x)) + " " +
    movies["cast"].apply(lambda x: " ".join(x)) + " " +
    movies["crew"]
)

movies["tags"] = movies["tags"].apply(lambda x: x.lower())

final_df = movies[["title", "tags"]]


# =========================
# VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
vectors = vectorizer.fit_transform(final_df["tags"]).toarray()


# =========================
# COSINE SIMILARITY
# =========================
similarity = cosine_similarity(vectors)


# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend(movie_name, top_n=5):
    if movie_name not in final_df["title"].values:
        print("❌ Movie not found in dataset")
        return

    index = final_df[final_df["title"] == movie_name].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:top_n+1]

    print(f"\n🎬 Recommendations for '{movie_name}':\n")
    for i in movies_list:
        print(final_df.iloc[i[0]].title)


# =========================
# SCRIPT EXECUTION
# =========================
if __name__ == "__main__":
    movie_name = input("Enter a movie name: ")
    recommend(movie_name)
