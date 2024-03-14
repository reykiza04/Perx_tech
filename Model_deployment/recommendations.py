import pandas as pd

# Location-Based Popularity Recommendation Model
def location_based_popularity_recommendation(users_df, merged_df_encoded, movies_df, zip_code):
    # Filter ratings by users in the specified zip code
    user_ids_in_zip_code = users_df[users_df['Zip-code'] == zip_code]['UserID']
    ratings_in_zip_code = merged_df_encoded[merged_df_encoded['UserID'].isin(user_ids_in_zip_code)]

    # Calculate the average rating for each movie in the specified zip code
    movie_avg_ratings_zip = ratings_in_zip_code.groupby('MovieID')['Rating'].mean()

    # Merge with movie information to obtain movie names and genres
    movie_info = movies_df[['MovieID', 'Title', 'Genres']]
    top_movies_zip = movie_avg_ratings_zip.to_frame(name='Average Rating').reset_index()
    top_movies_zip = pd.merge(top_movies_zip, movie_info, on='MovieID', how='left')

    # Sort movies by average rating in descending order
    top_movies_zip = top_movies_zip.sort_values(by='Average Rating', ascending=False).head(10)

    return top_movies_zip

# Overall Popularity-Based Recommendation Model
def overall_popularity_recommendation(merged_df_encoded, movies_df):
    # Calculate the average rating for each movie
    movie_avg_ratings = merged_df_encoded.groupby('MovieID')['Rating'].mean()

    # Merge with movie information to obtain movie names and genres
    movie_info = movies_df[['MovieID', 'Title', 'Genres']]
    top_movies_overall = movie_avg_ratings.to_frame(name='Average Rating').reset_index()
    top_movies_overall = pd.merge(top_movies_overall, movie_info, on='MovieID', how='left')

    # Sort movies by average rating in descending order
    top_movies_overall = top_movies_overall.sort_values(by='Average Rating', ascending=False).head(10)

    return top_movies_overall