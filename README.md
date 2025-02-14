# Emotion Recognition and Movie Recommendation

## Overview

The **Emotion Recognition and Movie Recommendation** application is an AI-driven platform designed to enhance users' movie-watching experience by providing personalized movie recommendations based on their emotions. By analyzing user-uploaded images, the system can detect emotions such as happiness, sadness, and anger, and suggest movies that match the detected mood. The application integrates secure Google OAuth-based login, allowing users to access and manage a list of their favorite movies, track recommendations, and revisit their past selections. The app's intuitive and responsive design is optimized for both desktop and mobile use, ensuring easy access for all users.

## Features

- **User Authentication**: Secure Google OAuth login for easy access.
- **Emotion Recognition**: Upload images or use a webcam to detect emotions.
- **Movie Recommendations**: Curated movie suggestions based on the detected emotion.
- **Favorites Management**: Save, view, and delete favorite movies for quick access.
- **Responsive Design**: Optimized for both desktop and mobile devices.

## Technologies Used

### Frontend

- **React**: JavaScript library for building user interfaces.
- **React Router**: Handles routing and navigation.
- **Axios**: Makes HTTP requests to the backend.
- **Bootstrap**: Styles components and ensures responsive design.
- **Webcam Component**: Captures images directly from the user’s camera.

### Backend

- **Flask**: Python web framework for building APIs.
- **TensorFlow**: Implements the emotion recognition model.
- **Google OAuth**: Secure user authentication.

## Installation

### Prerequisites

Ensure the following tools are installed before beginning:

- **Node.js** and **npm** for frontend development.
- **Python** and **pip** for backend development.

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
