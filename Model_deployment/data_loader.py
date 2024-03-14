import pandas as pd
# import joblib  # Assuming you're using joblib to save and load the model

def load_data():
    merged_df_encoded = pd.read_csv('merged_df_encoded.csv')
    movies_df = pd.read_csv('movies_df.csv')
    users_df = pd.read_csv('users_df.csv')
    return merged_df_encoded, movies_df, users_df

import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model('model_dl')  # Assuming your model is saved in the 'model_dl' folder
    return model