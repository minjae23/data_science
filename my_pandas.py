import pandas as pd
import numpy as np

n_arr1 = np.array([[3, '?', 2, 5], ['*', 4, 5, 6], ['+', 3, 2, '&'], [5, '?', 7, '!']])
df1 = pd.DataFrame(n_arr1)

print(df1)

n_arr2 = pd.DataFrame(n_arr1).apply(pd.to_numeric, errors='coerce')

df2 = pd.DataFrame(n_arr2)

print(df2)
#n_arr = pd.DataFrame(n_arr).replace({'?': np.nan, '*': np.nan, '+': np.nan, '&': np.nan, '!': np.nan})


print(df2.isna().any())
print(df2.isna().sum())
print(df2.dropna(axis= 0,how='all'))
print(df2.dropna(axis= 0,how='any'))

print(df2.dropna(axis= 0,thresh=1))
print(df2.dropna(axis= 0,thresh=2))

print(df2.fillna(100))
mean=  df2.mean()
print(df2.fillna(mean))
medium = df2.median()
print(df2.fillna(medium))

f_df = df2.ffill()
b_df = df2.bfill()

print(f_df)
print(b_df)