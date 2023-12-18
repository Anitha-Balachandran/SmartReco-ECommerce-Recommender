from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load models and data
classification_model = joblib.load('balanced_naive_bayes_model.pkl')
text_vectorizer = joblib.load('balanced_tfidf_vectorizer.pkl')
df = pd.read_csv('Cleaned_Products_Data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_query = request.form['user_query']
    query_tfidf = text_vectorizer.transform([user_query])
    predicted_category = classification_model.predict(query_tfidf)[0]

    filtered_df = df[df['Category'] == predicted_category]
    X = text_vectorizer.transform(filtered_df['summary_lemmatized'])
    target_tfidf_matrix = X

    query_similarity_scores = cosine_similarity(query_tfidf, target_tfidf_matrix)

    N = 6
    top_N_indices = np.argsort(query_similarity_scores[0])[-N:][::-1]

    columns_needed = ['title', 'description_cleaned', 'price', 'imageURL']
    top_N_recommendations = filtered_df.iloc[top_N_indices][columns_needed]

    # Exclude the user query from the recommendations
    # Assuming the 'title' column contains product titles
    user_query_lower = user_query.lower()
    top_N_recommendations = top_N_recommendations[top_N_recommendations['title'].str.lower() != user_query_lower]

    # Convert imageURL from string representation of list to actual list
    top_N_recommendations['imageURLHighRes'] = top_N_recommendations['imageURL'].apply(eval)

    recommendations = top_N_recommendations.to_dict(orient='records')
    return render_template('index.html', user_query=user_query, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
