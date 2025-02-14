from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MongoDB connection
client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017'))
db = client['favList']  # Use 'favList' as the database name
favorites_collection = db['favorites']  # Collection for favorite movies

# Load the trained model
model = tf.keras.models.load_model('emotion_recognition_model.h5')

# Update class labels to the new ones
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Load the movie dataset
movie_data = pd.read_csv('movie_dataset.csv')

# Define a mapping between emotions and genres (updated)
emotion_genre_mapping = {
    'happy': ['Comedy', 'Animation', 'Family'],
    'angry': ['Action', 'Thriller'],
    'disgust': ['Horror'],
    'fear': ['Horror', 'Thriller'],
    'neutral': ['Drama', 'Documentary'],
    'sad': ['Drama', 'Romance']
}

# Create uploads folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_emotion(img_path):
    """Predict the emotion from the uploaded image."""
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_labels[predicted_class_index]

def recommend_movies_by_emotion(emotion):
    """Recommend movies based on the predicted emotion."""
    genres = emotion_genre_mapping.get(emotion, [])
    if not genres:
        return pd.DataFrame()  # Return empty DataFrame if no genres are found
    
    # Filter movies based on genres
    recommended_movies = movie_data[movie_data['genre'].str.contains('|'.join(genres), case=False, na=False)]
    return recommended_movies[['title', 'director', 'release_year', 'genre', 'rating']]

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and predict the emotion."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Predict emotion and recommend movies
    try:
        predicted_emotion = predict_emotion(filepath)
        recommended_movies = recommend_movies_by_emotion(predicted_emotion)

        movie_list = recommended_movies.to_dict(orient='records') if not recommended_movies.empty else []
        return jsonify({
            'predicted_emotion': predicted_emotion,
            'recommended_movies': movie_list
        })
    except Exception as e:
        return jsonify({"error": "Error during prediction", "details": str(e)}), 500

@app.route('/add_favorite', methods=['POST'])
def add_favorite():
    """Add a favorite movie for a specific user."""
    data = request.json
    required_fields = ['title', 'director', 'release_year', 'genre', 'email']
    
    if not data or not all(field in data for field in required_fields):
        return jsonify({"error": "Invalid data"}), 400

    favorites_collection.insert_one(data)
    return jsonify({"message": "Movie added to favorites"}), 201

@app.route('/get_favorites', methods=['GET'])
def get_favorites():
    """Get favorite movies for a specific user."""
    email = request.args.get('email')
    if not email:
        return jsonify({"error": "Email not provided"}), 400

    favorites = list(favorites_collection.find({"email": email}, {'_id': 0}))
    return jsonify(favorites)

@app.route('/delete_favorite', methods=['DELETE'])
def delete_favorite():
    """Delete a favorite movie."""
    data = request.json
    if not data or not all(key in data for key in ['title', 'email']):
        return jsonify({"error": "Invalid data"}), 400
    
    result = favorites_collection.delete_one({"email": data['email'], "title": data['title']})
    if result.deleted_count > 0:
        return jsonify({"message": "Movie removed from favorites"}), 200
    else:
        return jsonify({"error": "Movie not found"}), 404

@app.route('/update_favorite', methods=['PUT'])
def update_favorite():
    """Update a favorite movie."""
    data = request.json
    if not data or not all(key in data for key in ['title', 'email']):
        return jsonify({"error": "Invalid data"}), 400
    
    result = favorites_collection.update_one(
        {"email": data['email'], "title": data['title']},
        {"$set": data}
    )
    if result.matched_count > 0:
        return jsonify({"message": "Favorite movie updated"}), 200
    else:
        return jsonify({"error": "Movie not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
