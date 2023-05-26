Bank Loan Prediction System using Machine Learning
This project aims to build and train a simple deep neural network model to predict the approval of personal loan for a person based on features like age, experience, income, locations, family, education, existing mortgage, credit card etc. The project uses Python programming language and various libraries such as pandas, numpy, seaborn, matplotlib, sklearn, keras and tensorflow.

Data Preparation
The project uses the historical dataset ‘Loan Eligible Dataset,’ available on Kaggle and licensed under Database Contents License (DbCL) v1.0. The dataset contains 1810 rows and 56 columns of various features and labels related to loan applicants. The dataset was processed and analyzed using Python programming libraries on Kaggle’s Jupyter Notebook cloud environment.

The following steps were performed to prepare the data for modeling:

Dropping null values columns: The columns with more than 50% missing values were dropped from the dataset. These include owner_2_score, owner_3_score, RATE_ID_FOR_fsr, RATE_ID_FOR_funded_last_30, INPUT_VALUE_ID_FOR_judgement_lien_time, RATE_ID_FOR_judgement_lien_time, RATE_ID_FOR_avg_net_deposits and RATE_ID_FOR_industry_type.
Using label encoder then KNN imputer: The categorical features were encoded using label encoder to convert them into numerical values. Then, the missing values in the numerical features were imputed using KNN imputer with 3 nearest neighbors.
Using pie chart to visualize distributions within columns: The distribution of each categorical feature was plotted using pie charts to understand the proportion of each category in the dataset.

Correlations with the Y column: The correlation between each numerical feature and the target variable (completion_status) was calculated using Pearson’s correlation coefficient. The features with high correlation (above 0.5 or below -0.5) were selected for further analysis.
Using VIF to check for multicollinearity: The variance inflation factor (VIF) was calculated for each selected feature to check for multicollinearity.
Using PCA (wrong but gave some insights): Principal component analysis (PCA) was applied to the remaining features to reduce the dimensionality and extract the most important components that explain the variance in the data. However, this was a wrong approach as PCA is not suitable for classification problems and it reduced the interpretability of the model.

Chi scores: The chi-square test was performed for each categorical feature and the target variable to check for independence. The features with low p-value (below 0.05) were selected as they indicate significant association with the target variable.

P scores: The p-value of each numerical feature and the target variable was calculated using and columns indicating high values were dropped.
Using box plots to check for outliers: The box plots of each numerical feature were plotted to check for outliers. The outliers were identified as the values that lie beyond 1.5 times the interquartile range (IQR) from the median.
Removing outliers using z-scores: The z-scores of each numerical feature were calculated to standardize them and remove the outliers. The outliers were defined as the values that have z-scores above 3 or below -3.
Imputing back the missing values after z scores using KNN imputer: The missing values that were created after removing the outliers were imputed back using KNN imputer with 3 nearest neighbors.

Model Building 
several models were used and the were optimized using grid search 

Results
Lgbm & XGBoost had the highrst accuracy of 96%
