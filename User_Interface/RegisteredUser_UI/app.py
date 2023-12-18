from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM

app = Flask(__name__)

# Load the LightFM model and the dataset
lightfm_model = joblib.load('logistic_lightfm.pkl')
df = pd.read_csv('reviews_2018.csv')

# Prepare the interaction matrix and store column labels
interaction_matrix_df = df.pivot_table(index='reviewerID', columns='asin', values='overall', fill_value=0)
column_labels = interaction_matrix_df.columns
sparse_interaction_matrix = csr_matrix(interaction_matrix_df)

# Generate mappings for user and item IDs
unique_users = df['reviewerID'].unique()
unique_items = df['asin'].unique()

user_dict = {user_id: i for i, user_id in enumerate(unique_users)}
item_dict = {
    'asin': {str(item_id): i for i, item_id in enumerate(unique_items)},
    'title': dict(zip(df['asin'], df['title'])),
    'price': dict(zip(df['asin'], df['price']))
}

def sample_recommendation_user(model, interactions, user_id, user_dict, item_dict, threshold=0, nrec_items=5, show=True):
    user_x = user_dict.get(user_id)
    if user_x is None:
        return []
    
    n_users, n_items = interactions.shape
    scores = pd.Series(model.predict(user_x, np.arange(n_items)))
    scores.index = column_labels  # Use column labels from the DataFrame

    user_interactions = interactions.getrow(user_x).toarray().flatten()
    known_items = np.where(user_interactions > threshold)[0]

    scores.iloc[known_items] = np.NINF  # Set scores of known items to negative infinity
    return_score_list = scores.nlargest(nrec_items).index.tolist()

    recommendations = []
    for item_asin in return_score_list:
        title = item_dict['title'].get(item_asin, "Unknown Item")
        price = item_dict['price'].get(item_asin, "Unknown Price")
        recommendations.append({"title": title, "price": price})

    return recommendations


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prodrecommend', methods=['POST'])
def prod_recommend():
    recommendations = []
    selected_user = request.form.get('user_id')
    print(f"Selected User ID: {selected_user}")
    #selected_user = 'A2GI8X394RLF83'
    if selected_user:
        try:
            recommendations = sample_recommendation_user(
                lightfm_model, 
                sparse_interaction_matrix, 
                user_id=selected_user, 
                user_dict=user_dict, 
                item_dict=item_dict, 
                show=False
            )
        except Exception as e:
            recommendations = [{"title": "Error", "price": str(e)}]
    print(f"Sending to template: {recommendations}")
    return render_template('index.html', user=selected_user, items=recommendations)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
