import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_excel("C:/Users/01062/Downloads/bmi_data_phw1.xlsx")

# DataFrame information
df_info = pd.DataFrame({
    'Feature Names': data.columns,
    'Data Types': data.dtypes
})

# Visualize the distribution of Height and Weight based on BMI categories
g = sns.FacetGrid(data, col="BMI")
g.map(plt.hist, "Height (Inches)", bins=10, color="blue")
g.map(plt.hist, "Weight (Pounds)", bins=10, color="red")
plt.show()

# Perform Linear Regression to predict BMI based on Height
X = data[['Height (Inches)']].values
y = data['BMI']
reg = LinearRegression().fit(X, y)

# Calculate the residuals
pred_w = reg.predict(data[['Height (Inches)']])
e = data["Weight (Pounds)"] - pred_w

# Standardize the residuals
standard_scaler = StandardScaler()
e_standard_scaled = standard_scaler.fit_transform(e.values.reshape(-1, 1))

# Plot the histogram of standardized residuals
plt.hist(e_standard_scaled, bins=10, color='blue', edgecolor='black')
plt.xlabel('Standardized Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Standardized Residuals')
plt.show()

# Define a threshold for outlier detection
std_threshold = 3

# Function to adjust BMI for outliers
def adjust_bmi(e, std_threshold):
    adjusted_bmi = []
    for error in e:
        if error > std_threshold:
            adjusted_bmi.append(4)  # Upper limit for outliers
        elif error < -std_threshold:
            adjusted_bmi.append(0)  # Lower limit for outliers
        else:
            adjusted_bmi.append(np.nan)  # Not an outlier
    return adjusted_bmi

# Adjust BMI for outliers
adjusted_bmi = adjust_bmi(e_standard_scaled, std_threshold)

# Convert 'Sex' column to binary (1 for Male, 0 for Female)
data['Sex'] = data['Sex'].apply(lambda x: 0 if x == 'Female' else 1)

# Split the data into Male and Female subsets
male_data = data.loc[data['Sex'] == 1, ['BMI']]
female_data = data.loc[data['Sex'] == 0, ['BMI']]

# Perform Linear Regression for Male and Female datasets separately
X_1 = male_data.index.values.reshape(-1, 1)
y_1 = male_data['BMI']
reg_1 = LinearRegression().fit(X_1, y_1)

X_2 = female_data.index.values.reshape(-1, 1)
y_2 = female_data['BMI']
reg_2 = LinearRegression().fit(X_2, y_2)

# Predict BMI for Male and Female datasets
pred_bmi_m = reg_1.predict(X_1)
pred_bmi_f = reg_2.predict(X_2)

# Convert predicted BMI values to Series with correct indexes
pred_bmi_m = pd.Series(pred_bmi_m, index=male_data.index)
pred_bmi_f = pd.Series(pred_bmi_f, index=female_data.index)

# Calculate residuals for Male and Female datasets
e_bmi_m = male_data["BMI"] - pred_bmi_m
e_bmi_f = female_data["BMI"] - pred_bmi_f

# Standardize the residuals
standard_scaler = StandardScaler()
e_bmi_standard_scaled_m = standard_scaler.fit_transform(e_bmi_m.values.reshape(-1, 1))
e_bmi_standard_scaled_f = standard_scaler.fit_transform(e_bmi_f.values.reshape(-1, 1))

# Plot histograms of standardized residuals for Male and Female datasets
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(e_bmi_standard_scaled_m, bins=10, color='red', edgecolor='black')
plt.xlabel('Standardized Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Standardized Residuals (Male)')

plt.subplot(1, 2, 2)
plt.hist(e_bmi_standard_scaled_f, bins=10, color='red', edgecolor='black')
plt.xlabel('Standardized Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Standardized Residuals (Female)')

plt.tight_layout()
plt.show()

# Calculate correlation coefficient between Sex and BMI
correlation_coefficient = data[['Sex', 'BMI']].corr().iloc[0, 1]
print("Correlation Coefficient between Sex and BMI:", correlation_coefficient)

# Print the statistical information of the dataset
print("Statistical Data:")
print(df_info)

# Print the feature names
print("\nFeature Names:")
print(data.columns)

# Print the data types of features
print("\nData Types:")
print(data.dtypes)
