import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
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

# # One-hot encode categorical variables
# X = pd.get_dummies(X, columns=categorical_columns, drop_first=True, dtype=int)

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
    'n_estimators': [50, 100, 150],  # Adjust the range of estimators as needed
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the Gradient Boosting Regressor model
gb_model = GradientBoostingRegressor()

# Perform Grid Search to find the best hyperparameters
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the Gradient Boosting Regressor model with the best hyperparameters
best_gb_model = GradientBoostingRegressor(n_estimators=best_params['n_estimators'],
                                          learning_rate=best_params['learning_rate'],
                                          max_depth=best_params['max_depth'],
                                          min_samples_split=best_params['min_samples_split'],
                                          min_samples_leaf=best_params['min_samples_leaf'])
best_gb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_best_gb = best_gb_model.predict(X_test)

# Evaluate the Gradient Boosting Regressor model with the best hyperparameters
mae_best_gb = mean_absolute_error(y_test, y_pred_best_gb)
mse_best_gb = mean_squared_error(y_test, y_pred_best_gb)

print("\nBest Gradient Boosting Regressor Results:")
print("Mean Absolute Error (MAE):", mae_best_gb)
print("Mean Squared Error (MSE):", mse_best_gb)

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['total gross'], bins=30, kde=True)
plt.title('Distribution of Total Gross')
plt.xlabel('Total Gross')
plt.ylabel('Frequency')
plt.show()

# Visualize the distribution of selected features
# for feature in selected_features:
#     plt.figure(figsize=(11, 7))
#     sns.histplot(df[feature], bins=30, kde=True)
#     plt.title(f'Distribution of {feature}')
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.xticks(rotation=45)
#     plt.show()

wordcloud = WordCloud(width=800, height=400, max_words=50, background_color='white').generate_from_frequencies(
    df['director'].value_counts())

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Plot the predicted values vs. true values
errors = abs(y_pred_best_gb - y_test)
cmap = plt.get_cmap('viridis')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_gb, c=errors, cmap=cmap)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()

# Scatter plot with diagonal line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_gb, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', linewidth=2)
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Total Gross')
plt.ylabel('Predicted Total Gross')
plt.show()
