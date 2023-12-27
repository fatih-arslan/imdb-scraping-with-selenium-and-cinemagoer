# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

# Load data from Excel file into a DataFrame
file_path = '../verbal_data.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Extract target variable
y = df['total gross']

# Drop target variable and non-numeric columns
X = df.drop(['total gross', 'title'], axis=1)  # Remove 'title' from features

# Convert categorical columns to 'category' data type
categorical_columns = ['content rating', 'genre', 'director', 'writer', 'producer', 'country', 'language']
X[categorical_columns] = X[categorical_columns].astype('category')

# Apply label encoding to categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

# Perform backward elimination for feature selection using statsmodels
X_ = sm.add_constant(X)
model = sm.OLS(y, X_).fit()
selected_features = list(X.columns)
p_values = list(model.pvalues)[1:]

while max(p_values) > 0.05:
    max_p_index = p_values.index(max(p_values))
    selected_features.pop(max_p_index)
    X = X[selected_features]
    X_ = sm.add_constant(X)
    model = sm.OLS(y, X_).fit()
    p_values = list(model.pvalues)[1:]

# Display the selected features after backward elimination
print("Selected Features:", selected_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'n_neighbors': [1, 2, 3, 5, 7, 9],  # Adjust the range of neighbors as needed
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Create the K-Nearest Neighbors Regression model
knn_model = KNeighborsRegressor()

# Perform Grid Search to find the best hyperparameters
grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the K-Nearest Neighbors Regression model with the best hyperparameters
best_knn_model = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'],
                                     weights=best_params['weights'],
                                     metric=best_params['metric'])

knn_model.fit(X_train, y_train)

# Train a K-Nearest Neighbors Regression model
# knn_model = KNeighborsRegressor(n_neighbors=1)  # You can adjust the number of neighbors as needed
# knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_knn = knn_model.predict(X_test)

# Evaluate the K-Nearest Neighbors Regression model
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)

print("\nK-Nearest Neighbors Results:")
print("Mean Absolute Error (MAE):", mae_knn)
print("Mean Squared Error (MSE):", mse_knn)

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['total gross'], bins=30, kde=True)
plt.title('Distribution of Total Gross (K-Nearest Neighbors)')
plt.xlabel('Total Gross')
plt.ylabel('Frequency')
plt.show()

# Scatter plot with diagonal line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_knn, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', linewidth=2)
plt.title('Actual vs. Predicted Values (K-Nearest Neighbors)')
plt.xlabel('Actual Total Gross')
plt.ylabel('Predicted Total Gross')
plt.show()
