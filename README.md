# SmartReco-ECommerce-Recommender
 A project revolutionizing E-Commerce experiences, SmartReco is a recommendation engine equipped with NLP, Naive Bayes Classifier, k-NN, Cosine Similarity, and LightFM for tailored user suggestions.

In the rapidly evolving e-commerce landscape, this project responds to the growing demand for personalized recommendation systems, specifically within the luxury beauty market. Leveraging advanced machine learning algorithms, the objective is to create a recommendation system that aligns with users' unique preferences, contributing to increased satisfaction and engagement. The project addresses challenges in the e-commerce sector by delivering personalized product recommendations through category classification, collaborative filtering, and the k-NN algorithm. Three distinct workflows cater to Guest Users, Registered Users, and both, ensuring inclusivity and overcoming the "cold start" problem. The project evaluates models for performance metrics, enhancing the overall shopping experience with relevant and context-aware recommendations. This innovative solution transforms the way users discover and engage with products in the luxury beauty e-commerce sector.


### Description: Developed recommendation systems for e-commerce, including "Products Similar to this item," "Top Picks for You," and "Customer who bought this item also bought."
<img width="760" alt="image" src="https://github.com/Anitha-Balachandran/SmartReco-ECommerce-Recommender/assets/143915040/cdebc8c3-4852-40f7-941c-db6d984b72c3">

## Products Similar to this item:

Algorithm: Ensemble Classification Model (XGBoost, SVM, Multinomial Naive Bayes, Random Forest, Logistic Regression)

Similarity Calculation: Cosine similarity on tokenized vectors from user query and product metadata (product_title, product_description)

Recommendation Process: Filters products by predicted category and ranks them based on similarity to user query

Differentiation: Utilizes category-specific filtering for enhanced relevance and precision compared to keyword-based search

Performance Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

<img width="1010" alt="image" src="https://github.com/Anitha-Balachandran/SmartReco-ECommerce-Recommender/assets/143915040/c43fc6dc-b048-431d-9c41-7853fb6f1e82">


## Top Picks for You:

Algorithm: LightFM (Collaborative filtering and content-based blending)

Matrix Factorization: Learns latent factors for users and items from the interaction matrix

Loss Optimization Functions: Includes Logistic, BPR, and WARP

Performance Metrics: Precision at k, AUC

<img width="1010" alt="image" src="https://github.com/Anitha-Balachandran/SmartReco-ECommerce-Recommender/assets/143915040/65ec9936-c95f-4802-a7ba-fb96dd4e7309">


## Customer who bought this item also bought:

Algorithm: k-Nearest Neighbors (kNN)

Similarity Calculation: Cosine similarity on tokenized vectors from 'also_bought' and 'also_viewed' product metadata

Recommendation Process: Recommends k most similar items based on item-item similarity

Benefit: Effective in sparse datasets or limited user-item interactions

<img width="987" alt="image" src="https://github.com/Anitha-Balachandran/SmartReco-ECommerce-Recommender/assets/143915040/017f27f7-da4e-4e41-8eac-9390a85cd74a">

#### Impact: This can be used to improve user engagement and satisfaction through personalized product recommendations based on user behavior and item similarities.

#### Results: Achieved high accuracy and precision in suggesting relevant products to users, enhancing the overall shopping experience.








