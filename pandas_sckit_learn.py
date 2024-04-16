import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesRegressor

# Load the California housing dataset
data = fetch_california_housing()

# Separate features and target variable
X, y = data.data, data.target

# Create a DataFrame from the features and target
df = pd.DataFrame(X, columns=data.feature_names)
df['Target'] = y

# Select the best K features using SelectKBest
bestfeatures = SelectKBest(score_func=f_regression, k=8)
fit = bestfeatures.fit(X, y)    
dfcolumns = pd.DataFrame(data.feature_names)
dfscores = pd.DataFrame(fit.scores_)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score'] # Name the dataframe columns
print(featureScores.nlargest(10, 'Score')) # Print the 10 best features

# Train an ExtraTreesRegressor model
model = ExtraTreesRegressor()
model.fit(X, y)
print(model.feature_importances_)

# Plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=data.feature_names)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# Calculate the correlation matrix
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
# Plot the heatmap
g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()
