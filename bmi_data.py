import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Load the dataset
data = pd.read_csv("C:/Users/01062/Downloads/bmi_data_lab2.csv")

# DataFrame information
df_info = pd.DataFrame({
    'Feature Names': data.columns,
    'Data Types': data.dtypes
})

# Function to scale data and plot histogram
def scale_and_plot_histogram(data, scaler, title):
    data = data[~np.isnan(data).any(axis=1)]  # Remove rows with NaN values
    scaled_data = scaler.fit_transform(data)
    plt.hist2d(scaled_data[:,0], scaled_data[:,1], bins=5, cmap='Blues')
    plt.colorbar(label='Counts')
    plt.title(title)
    plt.xlabel('Scaled Height')
    plt.ylabel('Scaled Weight')
    plt.grid(True)

# Extract Height and Weight data
height = data['Height (Inches)'].values
weight = data['Weight (Pounds)'].values
data_set = np.column_stack((height, weight))

# Scaling and plotting histograms using different scalers
scaler_std = StandardScaler()
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
scale_and_plot_histogram(data_set, scaler_std, 'StandardScaler')

scaler_minmax = MinMaxScaler()
plt.subplot(1, 3, 2)
scale_and_plot_histogram(data_set, scaler_minmax, 'MinMaxScaler')

scaler_robust = RobustScaler()
plt.subplot(1, 3, 3)
scale_and_plot_histogram(data_set, scaler_robust, 'RobustScaler')

plt.tight_layout()
plt.show()

# Handle NaN and outlier values
data.loc[(data['Height (Inches)'] < 30) | (data['Height (Inches)'] > 100), 'Height (Inches)'] = pd.NA
data.loc[(data['Weight (Pounds)'] < 50) | (data['Weight (Pounds)'] > 500), 'Weight (Pounds)'] = pd.NA

nan_count = data.isnull().sum()
total_rows_with_nan = data.isnull().any(axis=1).sum()

print("Number of rows with NaN:", total_rows_with_nan)
print("Number of NaN for each column:")
print(nan_count)

# Print the statistical data
print("Statistical Data:")
print(df_info)

# Print the feature names
print("\nFeature Names:")
print(data.columns)

# Print the data types
print("\nData Types:")
print(data.dtypes)

# Extract dirty records
dirty_records = data[
    (data['Height (Inches)'] < 30) | 
    (data['Height (Inches)'] > 100) | 
    (data['Weight (Pounds)'] < 50) | 
    (data['Weight (Pounds)'] > 500) | 
    (data['Height (Inches)'].isnull()) | 
    (data['Weight (Pounds)'].isnull())
]

# Extract dirty records by gender
dirty_records_male = data[ 
    (data['Sex'] == 'Male') & (
        (data['Height (Inches)'] < 30) | 
        (data['Height (Inches)'] > 100) | 
        (data['Weight (Pounds)'] < 50) | 
        (data['Weight (Pounds)'] > 500) | 
        (data['Height (Inches)'].isnull()) | 
        (data['Weight (Pounds)'].isnull())
    )
]

dirty_records_female = data[ 
    (data['Sex'] == 'Female') & (
        (data['Height (Inches)'] < 30) | 
        (data['Height (Inches)'] > 100) | 
        (data['Weight (Pounds)'] < 50) | 
        (data['Weight (Pounds)'] > 500) | 
        (data['Height (Inches)'].isnull()) | 
        (data['Weight (Pounds)'].isnull())
    )
]

# Extract dirty records by bmi
dirty_records_bmi1 = data[ 
    (data['BMI'] == 1) & (
        (data['Height (Inches)'] < 30) | 
        (data['Height (Inches)'] > 100) | 
        (data['Weight (Pounds)'] < 50) | 
        (data['Weight (Pounds)'] > 500) | 
        (data['Height (Inches)'].isnull()) | 
        (data['Weight (Pounds)'].isnull())
    )
]
dirty_records_bmi2 = data[ 
    (data['BMI'] == 2) & (
        (data['Height (Inches)'] < 30) | 
        (data['Height (Inches)'] > 100) | 
        (data['Weight (Pounds)'] < 50) | 
        (data['Weight (Pounds)'] > 500) | 
        (data['Height (Inches)'].isnull()) | 
        (data['Weight (Pounds)'].isnull())
    )
]
dirty_records_bmi3= data[ 
    (data['BMI'] == 3) & (
        (data['Height (Inches)'] < 30) | 
        (data['Height (Inches)'] > 100) | 
        (data['Weight (Pounds)'] < 50) | 
        (data['Weight (Pounds)'] > 500) | 
        (data['Height (Inches)'].isnull()) | 
        (data['Weight (Pounds)'].isnull())
    )
]
dirty_records_bmi = dirty_records.copy()
dirty_records_gender = dirty_records.copy()

# Likely-wrong value delete
data_cleaned = data.drop(dirty_records.index)
print(dirty_records)

# Create a clean dataset by removing rows with NaN values
clean_data = data_cleaned.dropna()

# Linear regression for predicting weight from height
X = clean_data[['Height (Inches)']]
y = clean_data['Weight (Pounds)']
reg = LinearRegression().fit(X, y)

# Linear regression for predicting height from weight
Y = clean_data[['Weight (Pounds)']]
x = clean_data['Height (Inches)']
reg_2 = LinearRegression().fit(Y, x)

# Iterate over dirty records and predict missing or outlier values for height and weight
for idx, row in dirty_records.iterrows():
    height = row['Height (Inches)']
    weight = row['Weight (Pounds)']
    
    # Perform prediction only if height or weight is NaN or an outlier
    if pd.isnull(height) or height < 30 or height > 100:
        predicted_height = reg_2.predict([[weight]])[0]
        dirty_records.at[idx, 'Height (Inches)'] = predicted_height
        data.at[idx, 'Height (Inches)'] = predicted_height  
      
    if pd.isnull(weight) or weight < 50 or weight > 500:
        predicted_weight = reg.predict([[height]])[0]
        dirty_records.at[idx, 'Weight (Pounds)'] = predicted_weight
        data.at[idx, 'Weight (Pounds)'] = predicted_weight

# Print dirty records and cleaned dataset
print(dirty_records)
print(data)

# Plot scatter plot with dirty records and cleaned data along with linear regression lines
plt.scatter(dirty_records['Height (Inches)'], dirty_records['Weight (Pounds)'], c='red', label='Dirty Records')
plt.scatter(data_cleaned['Height (Inches)'], data_cleaned['Weight (Pounds)'], c='blue', label='Clean Data')
plt.plot(X, reg.predict(X), color='black', linewidth=1, label='Linear Regression Line(height)')
plt.plot(reg_2.predict(Y),Y, color='green', linewidth=1, label='Linear Regression Line(weight)')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.title('Scatter Plot of Height vs Weight')
plt.legend()
plt.show()

# Visualize BMI distribution using histograms
g = sns.FacetGrid(data, col="BMI")
g.map(plt.hist, "Height (Inches)", bins=10, color="blue")
g.map(plt.hist, "Weight (Pounds)", bins=10, color="red")
plt.show()

# Map 'Sex' column to numerical values
clean_data['Sex'] = clean_data['Sex'].map({'Female': 0, 'Male': 1})
X_2 = clean_data[['Sex']]

# Linear regression for predicting weight and height based on gender
w_gender = clean_data['Weight (Pounds)']
reg_w_gender = LinearRegression().fit(X_2, w_gender)

h_gender = clean_data['Height (Inches)']
reg_h_gender = LinearRegression().fit(X_2, h_gender)

# Handling Dirty Values by Gender
for idx, row in dirty_records_gender.iterrows():
    height = row['Height (Inches)']
    weight = row['Weight (Pounds)']
    gender = row['Sex']
  
    # Perform prediction only if height or weight is NaN or an outlier
    if(gender == 'Female'):
        if pd.isnull(height) or height < 30 or height > 100:
            predicted_height = reg_h_gender.predict(X_2)[0]
            dirty_records_female.at[idx, 'Height (Inches)'] = predicted_height
        
        if pd.isnull(weight) or weight < 50 or weight > 500:
            predicted_weight = reg_w_gender.predict(X_2)[0]
            dirty_records_female.at[idx, 'Weight (Pounds)'] = predicted_weight
    elif(gender =='Male'):
        if pd.isnull(height) or height < 30 or height > 100:
            predicted_height = reg_h_gender.predict(X_2)[0]
            dirty_records_male.at[idx, 'Height (Inches)'] = predicted_height
        
        if pd.isnull(weight) or weight < 50 or weight > 500:
            predicted_weight = reg_w_gender.predict(X_2)[0]
            dirty_records_male.at[idx, 'Weight (Pounds)'] = predicted_weight

# Visualize dirty records by gender
plt.scatter(dirty_records_male['Height (Inches)'], dirty_records_male['Weight (Pounds)'], c='red', label='Dirty Records Male')
plt.scatter(dirty_records_female['Height (Inches)'], dirty_records_female['Weight (Pounds)'], c='gray', label='Dirty Records Female')
plt.scatter(data_cleaned['Height (Inches)'], data_cleaned['Weight (Pounds)'], c='blue', label='Clean Data')
plt.plot(reg_h_gender.predict(X_2),Y, color='green', linewidth=1, label='Linear Regression Line(weight)')
plt.plot(X, reg_w_gender.predict(X_2), color='black', linewidth=1, label='Linear Regression Line(height)')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.title('Scatter Plot of Height vs Weight')
plt.legend()
plt.show()

# Continue similar processing for BMI categories...

# Remove NaN values using forward fill and backward fill
data_without_nan = data_cleaned.dropna()

data_filled_ffill = data_cleaned.ffill()
data_filled_bfill = data_cleaned.bfill()


