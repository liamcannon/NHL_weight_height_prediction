import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

nhl_data = pd.read_csv('Hockey.csv', skiprows= 1)
nhl_bio = pd.read_csv('nhl_bio.csv')

# print(nhl_data.sample(10))


nhl_bio['first_name'] = nhl_bio['first_name'] + ' ' + nhl_bio['last_name']
nhl_bio.rename(columns={'first_name': 'Name'}, inplace=True)
nhl_bio = nhl_bio.drop(['last_name'], axis=1)
# print(nhl_bio.sample(10))
# ignore the first row
nhl_output = pd.merge(nhl_data, nhl_bio, on='Name',how='left')

nhl_output=nhl_output.dropna(subset=['weight','active'])

print(nhl_output.sample(10))