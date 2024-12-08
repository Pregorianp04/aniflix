from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Daftar genre
genres = ["Action", "Comedy", "Kids", "Hentai", "School", "Adventure", "Fantasy", "Romance", "Drama"]

# Import dataset
ratings = pd.read_csv('data/ratings.csv')  
animes = pd.read_csv('data/anime.csv')  

# Data preparation
ratings = ratings.dropna()
animes = animes.dropna()
ratings = pd.merge(animes, ratings, left_on='anime_id', right_on='anime_id')
user_item_matrix = ratings.pivot_table(index='userId', columns='name', values='rating').fillna(0)
user_item_sparse = csr_matrix(user_item_matrix)
user_similarity = cosine_similarity(user_item_sparse)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def filter_by_genre(animes, preferred_genres):
    filtered_animes = animes[animes['genre'].str.contains('|'.join(preferred_genres), case=False, na=False)]
    return filtered_animes

def recommend_anime(user_id, preferred_genres, num_recommendations=10):
    genre_filtered_animes = filter_by_genre(animes, preferred_genres)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:]
    similar_users_ratings = user_item_matrix.loc[similar_users.index]
    weighted_ratings = similar_users_ratings.T.dot(similar_users)
    recommendation_scores = weighted_ratings / similar_users.sum()
    unrated_animes = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 0].index
    recommendations = recommendation_scores[unrated_animes].sort_values(ascending=False)
    recommendations = recommendations[recommendations.index.isin(genre_filtered_animes['name'])]
    return recommendations.head(num_recommendations)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Ambil genre dan rating dari form
            selected_genres = request.form.getlist('genres[]')
            rating = int(request.form.get('rating'))  
            user_id = 1  # Contoh user_id, bisa diganti sesuai implementasi
            
            # Dapatkan rekomendasi
            recommendations = recommend_anime(user_id, selected_genres)
            return render_template('index.html', genres=genres, recommendations=recommendations.index.tolist())
        except Exception as e:
            return render_template('index.html', genres=genres, error=str(e))
    
    # GET request (pertama kali buka halaman)
    return render_template('index.html', genres=genres)

if __name__ == '__main__':
    app.run(debug=True)
