from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import ast

app = Flask(__name__)




import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Function to read CSV with different encodings
def read_csv_with_encoding(file_path, encodings):
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file {file_path} with provided encodings.")

# Reading the CSV
encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1']
df = read_csv_with_encoding('cleaned_final_product.csv', encodings_to_try)

# Ensure nltk data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Tokenization and lemmatization functions
def tokenize_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return tokens

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Apply tokenization and lemmatization
df['summary_tokens'] = df['summary_cleaned'].apply(tokenize_text)
df['summary_lemmatized'] = df['summary_tokens'].apply(lemmatize_text)

def get_recommendations(selected_asin, n_neighbors=5):
    selected_product = df[df['asin'] == selected_asin]
    
    # Extract relevant information
    also_buy_list = ast.literal_eval(selected_product['also_buy'].values[0])
    also_view_list = ast.literal_eval(selected_product['also_view'].values[0])

    # Filter DataFrame and create a copy
    filtered_products = df[((df['asin'].isin(also_buy_list)) | 
                            (df['asin'].isin(also_view_list)) | 
                            (df['asin'] == selected_asin))].copy()

    filtered_products['summary_lemmatized_text'] = filtered_products['summary_lemmatized'].apply(' '.join)
    filtered_products.reset_index(drop=True, inplace=True)

    # Ensure n_neighbors is not greater than the number of samples
    n_samples = len(filtered_products)
    if n_samples <= n_neighbors:
        n_neighbors = n_samples - 1  # Subtract 1 to exclude the product itself

    if n_neighbors < 1:
        return []  # Not enough data to generate recommendations

    # Apply TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(filtered_products['summary_lemmatized_text'])

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(X)

    # Get the index of the selected product
    selected_index = filtered_products[filtered_products['asin'] == selected_asin].index[0]

    # Find the indices of the nearest neighbors
    distances, neighbor_indices = knn.kneighbors(X[selected_index:selected_index+1], n_neighbors=n_neighbors+1)

    # Exclude the selected product itself
    neighbor_indices = neighbor_indices[0][1:]

    # Retrieve the ASINs of the nearest neighbors
    nearest_neighbor_asins = filtered_products.iloc[neighbor_indices]['asin'].tolist()

    # Sort nearest neighbors
    nearest_neighbors_sorted = df[df['asin'].isin(nearest_neighbor_asins)].sort_values(by='overall', ascending=False)

    # Convert the DataFrame to a list of dictionaries
    recommendations_list = nearest_neighbors_sorted[['asin', 'title', 'price']].head(n_neighbors).to_dict(orient='records')

    return recommendations_list


@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_products = None
    selected_asin = None

    if request.method == 'POST':
        selected_asin = request.form.get('product_asin')
        recommended_products = get_recommendations(selected_asin)

    return render_template('index.html', products=df, recommended_products=recommended_products, selected_asin=selected_asin)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
