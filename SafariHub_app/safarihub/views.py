from django.http import HttpResponse
from django.shortcuts import render
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load model, vectorizer, and scaler
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

models_dir = os.path.join(base_dir, 'Recommendation_models')
app_data_dir = os.path.join(base_dir, 'App_data')


# Define paths for models, vectorizers, and scalers
model_paths = {
    'attraction': os.path.join(models_dir, 'destination-recommender.joblib'),
    'hotel': os.path.join(models_dir, 'hotel-recommender.joblib'),
    'tourops': os.path.join(models_dir, 'tourops-recommender.joblib'),
}
vectorizer_paths = {
    'attraction': os.path.join(models_dir, 'att_vectorizer.joblib'),
    'hotel': os.path.join(models_dir, 'hot_vectorizer.joblib'),
    'tourops': os.path.join(models_dir, 'opr_vectorizer.joblib'),
}
scaler_paths = {
    'attraction': os.path.join(models_dir, 'att_scaler.joblib'),
    'hotel': os.path.join(models_dir, 'hot_scaler.joblib'),
    'tourops': os.path.join(models_dir, 'opr_scaler.joblib'),
}

# Load models, vectorizers, and scalers
models = {k: joblib.load(v) for k, v in model_paths.items()}
vectorizers = {k: joblib.load(v) for k, v in vectorizer_paths.items()}
scalers = {k: joblib.load(v) for k, v in scaler_paths.items()}


# Load datasets
attraction_data_path = os.path.join(app_data_dir, 'attraction_data.json')
hotel_data_path = os.path.join(app_data_dir, 'hotel_data.json')
tours_data_path = os.path.join(app_data_dir, 'tours_data.json')
data_path = os.path.join(app_data_dir, 'data.json')
attraction_data = pd.read_json(attraction_data_path)
hotel_data = pd.read_json(hotel_data_path)
tours_data = pd.read_json(tours_data_path)
data = pd.read_json(data_path)

# Function to process the datasets and split them
def prepare_data(df, vectorizer, scaler, bigram_col='flattened_bigrams', label_col='similar', fit_scaler=False):
    # Define expected columns based on data type
    if df.equals(attraction_data):
        expected_columns = ['category_encoded', 'rating', 'numberOfReviews', 'photoCount', 'adjusted_sentiment',
                            'location_encoded', 'province_encoded', 'priceLevelencoded', label_col]
    elif df.equals(hotel_data):
        expected_columns = ['category_encoded', 'rating', 'numberOfReviews', 'photoCount', 'adjusted_sentiment',
                            'location_encoded', 'province_encoded', 'priceLevelencoded', 'upperPrice', 'lowerPrice', label_col]
    elif df.equals(tours_data):
        expected_columns = ['category_encoded', 'rating', 'numberOfReviews', 'photoCount', 'adjusted_sentiment',
                            'location_encoded', 'province_encoded', 'priceLevelencoded', label_col]
    else:
        raise ValueError("Unsupported data_type provided")

    # Ensure all expected columns are present
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {', '.join(missing_columns)}")

    X = df[expected_columns]

    # Apply vectorization to the bigram column
    bigram_matrix = vectorizer.transform(df[bigram_col])
    
    # Combine features
    combined_features = np.hstack((X, bigram_matrix.toarray()))

    # Fit the scaler if specified
    if fit_scaler:
        scaler.fit(combined_features)

    # Transform the combined features
    X_scaled = scaler.transform(combined_features)

    return X_scaled, df['name'], df[label_col]

# Prepare data for each dataset using corresponding vectorizers and scalers
X_train_att, names_train_att, y_train_att = prepare_data(attraction_data, vectorizers['attraction'], scalers['attraction'], fit_scaler=True)
X_train_hot, names_train_hot, y_train_hot = prepare_data(hotel_data, vectorizers['hotel'], scalers['hotel'], fit_scaler=True)
X_train_ops, names_train_ops, y_train_ops = prepare_data(tours_data, vectorizers['tourops'], scalers['tourops'], fit_scaler=True)

image_dict = data.set_index('name')['image'].to_dict()

# Ensure images are mapped to attraction, hotel, and tours data
attraction_data['image'] = attraction_data['name'].map(image_dict)
hotel_data['image'] = hotel_data['name'].map(image_dict)
tours_data['image'] = tours_data['name'].map(image_dict)

# Convert to records for rendering
attractions = attraction_data.to_dict(orient='records')
hotels = hotel_data.to_dict(orient='records')
tours = tours_data.to_dict(orient='records')

def index(request):

    return render(request, 'index.html', {
        'attractions': attractions,
        'hotels': hotels,
        'tours': tours,
    })


def recommend_items(model, data, names, item_name, category, top_n=5):
    item_idx = names[names == item_name].index[0]
    distances, indices = model.kneighbors([data[item_idx]], n_neighbors=top_n+1)
    recommended_indices = indices.flatten()[1:]

    # Fetch recommended items from the original DataFrame
    recommended_names = names.iloc[recommended_indices].values
    
    # Select the appropriate DataFrame based on category
    if category == 'attraction':
        recommended_data = attraction_data[attraction_data['name'].isin(recommended_names)]
        recommendations = recommended_data[['name', 'rating', 'location', 'image']].to_dict('records')
    elif category == 'hotel':
        recommended_data = hotel_data[hotel_data['name'].isin(recommended_names)]
        recommendations = recommended_data[['name', 'rating', 'priceRange', 'priceLevel', 'image']].to_dict('records')
    elif category == 'tour operator':
        recommended_data = tours_data[tours_data['name'].isin(recommended_names)]
        recommendations = recommended_data[['name', 'rating', 'numberOfReviews', 'main_bigram', 'image']].to_dict('records')
    else:
        recommendations = []

    return recommendations



def attraction_detail(request, attraction_name):
    recommendations = recommend_items(models['attraction'], X_train_att, names_train_att, attraction_name, category='attraction')
    return render(request, 'detail_attraction.html', {'recommendations': recommendations})

def hotel_detail(request, hotel_name):
    recommendations = recommend_items(models['hotel'], X_train_hot, names_train_hot, hotel_name, category='hotel')
    return render(request, 'detail_hotel.html', {'recommendations': recommendations})

def tourop_detail(request, tourop_name):
    recommendations = recommend_items(models['tourops'], X_train_ops, names_train_ops, tourop_name, category='tour operator')
    return render(request, 'detail_tourop.html', {'recommendations': recommendations})


