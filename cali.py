import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA


# Load the California housing dataset
data = fetch_california_housing()

# Separate features and target variable
X, y = data.data, data.target

# Create a DataFrame from the features and target
df = pd.DataFrame(X, columns=data.feature_names)
df['Target'] = y

# Apply PCA with 2 principal components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

# Create a DataFrame for the principal components
principalDf = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'])

# Combine the principal components DataFrame with the target variable DataFrame
finalDf = pd.concat([principalDf, df[['Target']]], axis=1)

# Plot the 2-component PCA
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

# Get unique target values and define colors for each target
targets = np.unique(y)
colors = ['r', 'g', 'b']

# Plot each target with corresponding color
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'Principal Component 1'],
               finalDf.loc[indicesToKeep, 'Principal Component 2'], 
               c=color, s=50)

# Add legend and grid
ax.legend(targets)
ax.grid()