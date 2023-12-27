# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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

# Support Vector Machine Regression
svm_model = SVR(kernel='linear')  # You can try different kernels such as 'rbf', 'poly', etc.
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
mae_svm = mean_absolute_error(y_test, y_pred_svm)
mse_svm = mean_squared_error(y_test, y_pred_svm)

print("\nSupport Vector Machine Results:")
print("Mean Absolute Error (MAE):", mae_svm)
print("Mean Squared Error (MSE):", mse_svm)

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['total gross'], bins=30, kde=True)
plt.title('Distribution of Total Gross (Support Vector Machine)')
plt.xlabel('Total Gross')
plt.ylabel('Frequency')
plt.show()

# Visualize the distribution of selected features
# for feature in selected_features:
#     plt.figure(figsize=(11, 7))
#     sns.histplot(df[feature], bins=30, kde=True)
#     plt.title(f'Distribution of {feature} (Support Vector Machine)')
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.xticks(rotation=45)
#     plt.show()

# Plot the predicted values vs. true values
errors = abs(y_pred_svm - y_test)
cmap = plt.get_cmap('viridis')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_svm, c=errors, cmap=cmap)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()

# Scatter plot with diagonal line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_svm, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', linewidth=2)
plt.title('Actual vs. Predicted Values (Linear Regression)')
plt.xlabel('Actual Total Gross')
plt.ylabel('Predicted Total Gross')
plt.show()
