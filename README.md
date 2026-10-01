# House Price Prediction - Regression Modeling (Ames Housing Dataset)

End-to-end regression project on the Kaggle Ames Housing dataset, predicting final sale price from house features.

## Problem Statement

Given a set of house features (size, quality, location-related attributes, age, condition, etc.), predict the final sale price as accurately as possible. This is a supervised regression problem evaluated with the Kaggle House Prices competition's train/test split.


## Project Overview
Combined train and test sets (2,919 rows, about 80 features) for consistent cleaning and encoding.
Handled missing values with feature-specific strategies: basement, garage, and lot frontage fields were each imputed differently based on what a missing value means for that feature.
Converted misclassified numeric columns (year and month fields) to categorical types where appropriate.
Encoded ordinal quality and condition features (basement condition, exposure, finish type) as ordered categories.
Corrected skewed numeric features with log transforms, and one-hot encoded the remaining categorical features.
Scaled features with RobustScaler before modeling.

## Tools Used
pandas, numpy
matplotlib, seaborn for EDA, correlation heatmaps, and skewness plots
scikit-learn for preprocessing, model training, and cross-validation
XGBoost

## Models Trained and Cross-Validated
Linear Regression, Ridge, Lasso, Polynomial Regression, SVR, Decision Tree Regressor, Random Forest Regressor, Bagging Regressor, Gradient Boosting Regressor, and XGBoost, each evaluated with K-fold cross-validation (R2).

## Results

Each model was scored with 3-fold cross-validation (R2, log-transformed SalePrice target):

| Model | R2 |
|---|---|
| Linear Regression | 0.760 |
| Ridge | 0.865 |
| Lasso | 0.868 |
| Polynomial Regression | 0.826 |
| SVR | 0.847 |
| Decision Tree | 0.674 |
| Random Forest | 0.856 |
| Bagging | 0.857 |
| Gradient Boosting | 0.883 |

Gradient Boosting scored highest in cross-validation (R2 = 0.883). SVR (R2 = 0.847) was used to generate the final predictions and was saved as the deployed model; a 10-fold cross-validation check on Linear Regression separately confirmed a consistent mean R2 of 0.802, supporting that the cross-validation setup was stable across folds.


## Key Steps
Correlation analysis to identify features most predictive of sale price.
Missing-value imputation tailored to each feature group rather than one blanket strategy.
Log-transforming skewed features to improve model performance.
Final model (SVR) used to generate predictions on the held-out test set and saved with pickle.

## Data Source

Ames Housing dataset via Kaggle's House Prices: Advanced Regression Techniques competition. Used for educational and analytical purposes only.

## Author

Rachna Kandari
kandari.rachna74@gmail.com | https://www.linkedin.com/in/rachna-kandari/
