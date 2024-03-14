from flask import Flask, render_template
from data_loader import load_data, load_model  # Import the load_model function
from recommendations import location_based_popularity_recommendation, overall_popularity_recommendation
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    # Load data and model
    merged_df_encoded, movies_df, users_df = load_data()
    
    model = load_model()  # Load the deep learning model
    # Example usage of recommendation models
    overall_recommendations = overall_popularity_recommendation(merged_df_encoded, movies_df)
    zip_code = '12345'  # Example zip code
    location_recommendations = location_based_popularity_recommendation(users_df, merged_df_encoded, movies_df, zip_code)
    # Assuming merged_df_encoded has been loaded and contains genre columns
    genre_columns = ['Drama', 'Animation', "Children's", 'Musical', 'Romance', 
                    'Comedy', 'Action', 'Adventure', 'Fantasy', 'Sci-Fi', 
                    'War', 'Thriller', 'Crime', 'Mystery', 'Western', 
                    'Horror', 'Film-Noir', 'Documentary']

    # Perform model predictions for the user
    user_id = 8
    user_movies = merged_df_encoded[merged_df_encoded['UserID'] == user_id]['MovieID'].unique()
    user_unrated_movies = merged_df_encoded[~merged_df_encoded['MovieID'].isin(user_movies)]['MovieID'].unique()
    user_recommendations_dict = make_predictions(model, user_id, user_unrated_movies, genre_columns)

    # Convert DataFrames to dictionaries
    overall_recommendations_dict = overall_recommendations.to_dict(orient='records')
    location_recommendations_dict = location_recommendations.to_dict(orient='records')
    
    return render_template('index.html', location_recommendations=location_recommendations_dict, overall_recommendations=overall_recommendations_dict, user_recommendations=user_recommendations_dict, zip_code=zip_code, user_id=user_id)
def make_predictions(model, user_id, user_unrated_movies, genre_columns):
    # Load data
    
    # Load data and model
    merged_df_encoded, movies_df, users_df = load_data()
    model = load_model()  # Load the deep learning model
       
    user_movies = merged_df_encoded[merged_df_encoded['UserID'] == user_id]['MovieID'].unique()
    # Filter out movies already rated by the user
    user_movies = merged_df_encoded[merged_df_encoded['UserID'] == user_id]['MovieID'].unique()
    user_unrated_movies = merged_df_encoded[~merged_df_encoded['MovieID'].isin(user_movies)]['MovieID'].unique()
    
    # Create input data for prediction
    user_ids = np.array([user_id] * len(user_unrated_movies))
    genre_input_data = np.zeros((len(user_unrated_movies), len(genre_columns)))
    
    # Predict ratings for unrated movies
    user_recommendations = model.predict([user_ids.reshape(-1, 1), user_unrated_movies.reshape(-1, 1), genre_input_data])
    
    # Create DataFrame with movie IDs and recommendation scores
    user_recommendations_df = pd.DataFrame({'MovieID': user_unrated_movies, 'Recommendation Score': user_recommendations.flatten()})
    user_recommendations_df = user_recommendations_df.sort_values(by='Recommendation Score', ascending=False)
    
    # Merge recommendations DataFrame with movies DataFrame
    recommended_movies_df = pd.merge(user_recommendations_df, movies_df, on='MovieID', how='left')
    
    # Convert DataFrame to dictionary
    user_recommendations_dict = recommended_movies_df.to_dict(orient='records')
    
    return user_recommendations_dict



if __name__ == '__main__':
    app.run(debug=True)
