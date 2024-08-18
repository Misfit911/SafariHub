import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Placeholder for data loading function
def load_data():
    return pd.read_csv("attraction_data.csv"), pd.read_csv("hotel_data.csv"), pd.read_csv("tour_data.csv")

# Load data
attraction_data, hotel_data, tour_data = load_data()

# Feature preprocessing and model training
def preprocess_and_train(data, bigram_column):
    # Check for and handle NaN values in the bigram_column silently
    if data[bigram_column].isnull().any():
        data[bigram_column] = data[bigram_column].fillna('')
    
    data['similar'] = ((data['rating'].diff().abs() < 0.5) & 
                       (data['priceLevelencoded'].diff().abs() == 0)).astype(int)
    
    names = data['name']
    y = data['similar']
    
    # Define features based on the dataset
    if bigram_column == 'main_bigram':  # For tour data
        X = data[['category_encoded', 'rating', 'numberOfReviews', 'photoCount', 
                  'adjusted_sentiment', 'location_encoded', 'province_encoded', 
                  'priceLevelencoded', 'similar']]
    else:  # For attraction and hotel data
        X = data[['category_encoded', 'rating', 'numberOfReviews', 'photoCount', 
                  'adjusted_sentiment', 'location_encoded', 'province_encoded', 
                  'priceLevelencoded', 'similar']]
    
    vectorizer = CountVectorizer()
    bigram_matrix = vectorizer.fit_transform(data[bigram_column])
    
    combined_features = np.hstack((X, bigram_matrix.toarray()))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined_features)
    
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X_scaled, y, names, test_size=0.2, random_state=42)
    
    knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn_model.fit(X_train)
    
    return knn_model, X_train, y_train, names_train

# Train models for each data type with appropriate bigram column
knn_attractions, X_train_attractions, y_train_attractions, names_train_attractions = preprocess_and_train(attraction_data, 'flattened_bigrams')
knn_hotels, X_train_hotels, y_train_hotels, names_train_hotels = preprocess_and_train(hotel_data, 'flattened_bigrams')
knn_tours, X_train_tours, y_train_tours, names_train_tours = preprocess_and_train(tour_data, 'main_bigram')

# Function to get recommendations
def recommend(knn_model, X_train, names_train, item_name, top_n=5):
    idx = names_train[names_train == item_name].index[0]
    distances, indices = knn_model.kneighbors([X_train[idx]], n_neighbors=top_n+1)
    recommended_names = names_train.iloc[indices.flatten()[1:]].values
    return recommended_names

# Streamlit UI
st.title("SafariHub Tour Recommendation System")

# Sidebar for selection
st.sidebar.title("Choose Your Experience")
options = st.sidebar.selectbox("Select the type of recommendation you are interested in:", 
                               ("Attractions", "Hotels", "Tours"))

# Handle user selection
if options == "Attractions":
    selected_attraction = st.sidebar.selectbox("Select an attraction:", names_train_attractions)
    if st.sidebar.button("Get Recommendations"):
        recommended_attractions = recommend(knn_attractions, X_train_attractions, names_train_attractions, selected_attraction)
        st.write(f"Recommended Attractions for {selected_attraction}:")
        for i, name in enumerate(recommended_attractions):
            st.write(f"{i+1}. {name}")
elif options == "Hotels":
    selected_hotel = st.sidebar.selectbox("Select a hotel:", names_train_hotels)
    if st.sidebar.button("Get Recommendations"):
        recommended_hotels = recommend(knn_hotels, X_train_hotels, names_train_hotels, selected_hotel)
        st.write(f"Recommended Hotels for {selected_hotel}:")
        for i, name in enumerate(recommended_hotels):
            st.write(f"{i+1}. {name}")
elif options == "Tours":
    selected_tour = st.sidebar.selectbox("Select a tour:", names_train_tours)
    if st.sidebar.button("Get Recommendations"):
        recommended_tours = recommend(knn_tours, X_train_tours, names_train_tours, selected_tour)
        st.write(f"Recommended Tours for {selected_tour}:")
        for i, name in enumerate(recommended_tours):
            st.write(f"{i+1}. {name}")

# Customize the styling of the app
st.markdown("""
    <style>
    .stTitle {font-size: 24px; color: #2e7bcf; font-weight: bold;}
    .stSidebar {background-color: #f4f4f4;}
    .stSidebar select {font-size: 16px; color: #000;}
    </style>
    """, unsafe_allow_html=True)
