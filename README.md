# Donor Proposal Approval Prediction with Naive Bayes

## Project Overview
This project applies **Multinomial Naive Bayes** to the DonorsChoose dataset to predict the approval status of grant proposals. By leveraging both categorical and numerical features, as well as text encoding for essay content, the model aims to provide high accuracy in predicting approvals.


- **Data Splitting**:
  - For **GridSearchCV** or **RandomSearchCV**: These methods incorporate k-fold cross-validation, so a simple `X_train` and `X_test` split is sufficient.
  - For manual hyperparameter tuning with loops, split the data into `X_train`, `X_cv`, and `X_test`.
  - **Stratified Splitting**: Ensures balanced class distribution in training and testing splits.

## Features
Key features are processed across two sets with different encoding techniques for essay content:

### Categorical Features
- **teacher_prefix**
- **project_grade_category**
- **school_state**
- **clean_categories**
- **clean_subcategories**

### Numerical Features
- **price**
- **teacher_number_of_previously_posted_projects**

### Essay Content Encoding
- Preprocessed and encoded essay content, experimenting with **max_features** and **n_grams** parameters to potentially increase AUC scores.

## Feature Sets
1. **Set 1**: Categorical and Numerical Features + Essay encoded with **Bag of Words (BOW)**
2. **Set 2**: Categorical and Numerical Features + Essay encoded with **TF-IDF**

## Methodology

### Hyperparameter Tuning
- **Alpha (smoothing parameter)**: Explored a range from `10^-5` to `10^2` (e.g., `[0.00001, 0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]`).
- **Class Prior**: Tested with values such as `[0.5, 0.5]` to observe how it affects predictions.

### Model Training
- Used **GridSearchCV** or **RandomSearchCV** for k-fold cross-validation during hyperparameter tuning to find the best alpha value that maximizes AUC.

### Model Evaluation
1. **AUC and ROC Curves**: After identifying optimal hyperparameters, trained the model and evaluated using AUC and ROC curves on both the training and test sets.
2. **Performance Visualization**:
   - Plotted model performance for each alpha value (log scale on the x-axis for readability).
   - Generated ROC curves for the final model on both training and test sets.
3. **Confusion Matrix**: Plotted the confusion matrix as a heatmap, with predicted vs. actual labels on the test data.

## Top Features Analysis
Used the `feature_log_prob_` parameter in **MultinomialNB**, identified and displayed the top 20 features for the chosen feature set (either Set 1 or Set 2), capturing both positive and negative influences. 

## Results Summary
Summarized the final model performance, including:
- **Optimal Hyperparameters**: Alpha and class_prior values
- **AUC Scores**: For both training and test data
- **Top Features**: Displayed top positive and negative features contributing to model predictions

