import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
# from pandas import dummies
nhl_data = pd.read_csv('Hockey.csv', skiprows= 1)
nhl_bio = pd.read_csv('nhl_bio.csv')

# print(nhl_data.sample(10))

nhl_bio['first_name'] = nhl_bio['first_name'] + ' ' + nhl_bio['last_name']
nhl_bio.rename(columns={'first_name': 'Name'}, inplace=True)
nhl_bio = nhl_bio.drop(['last_name'], axis=1)
# print(nhl_bio.sample(10))
# ignore the first row
nhl_output = pd.merge(nhl_data, nhl_bio, on='Name',how='left')

nhl_output[~nhl_output['weight'].notnull() | ~nhl_output['active'].notnull()].to_csv('nhl_output_missing.csv', index=False)

nhl_output=nhl_output.dropna(subset=['weight','active'])

nhl_output.to_csv('nhl_output.csv', index=False)

nhl_output.sample(10)

test = nhl_bio[['weight','height']]

def parse_height(height):
    split = height.split("'")
    feet = int(split[0].strip())
    inches = int(split[1].strip().replace('"', '').strip())
    return feet * 12 + inches

def parse_position(position):
    # parses the position of the player into a 0 or 1
    # 0 = forward
    # 1 = defense
    # 2 = goalie
    if( position == 'F'):
        return 0
    if( position == 'D'):
        return 1
    else:
        return 2
def parse_time(time):
    # parses the time on ice into a float
    # convert the time to a string
    # if time is NAN return 0
    # print(time, type(time))
    if pd.isnull(time):
        return 0
    split = time.split(':')
    minutes = int(split[0].strip())
    seconds = int(split[1].strip())
    # return a float
    return minutes + seconds/60

def parse_nan(value, column:str):
    # if the value is Nan return the median value of the column
    if pd.isnull(value):
        return nhl_output[column].median()
    else:
        return value

# if the column contains Nan values print the column name
print(nhl_output.columns[nhl_output.isna().any()].tolist())

# not the optimal soltuon but it works
# should've clusted the data and then used the median of the cluster

nhl_output['HITS'] = nhl_output['HITS'].apply(parse_nan, column = 'HITS')    
nhl_output['BS'] = nhl_output['BS'].apply(parse_nan, column = 'BS')
nhl_output['PP'] = nhl_output['PP'].apply(parse_time)
nhl_output['TOI'] = nhl_output['TOI'].apply(parse_time)
nhl_output['Pos'] = nhl_output['Pos'].apply(parse_position)
nhl_output['height'] = nhl_output['height'].apply(parse_height)
test['height'] = test['height'].apply(parse_height)

plt.scatter(test['weight'], test['height'])
plt.xlabel('weight (lbs)')
plt.ylabel('height (inches)')

kmeans = KMeans(n_clusters=2, random_state=42)
y_pred = kmeans.fit_predict(test[['height', 'weight']])

# get the median of each cluster
cluster_0_median = test[y_pred == 0].median()
cluster_1_median = test[y_pred == 1].median()

cluster_0_median_weight = cluster_0_median['weight']
cluster_0_median_height = cluster_0_median['height']/12

cluster_1_median_weight = cluster_1_median['weight']
cluster_1_median_height = cluster_1_median['height']/12

# print cluster values
print('Cluster 0 median weight: ' + str(cluster_0_median_weight))
print('Cluster 0 median height: ' + str(cluster_0_median_height))
print('Cluster 1 median weight: ' + str(cluster_1_median_weight))
print('Cluster 1 median height: ' + str(cluster_1_median_height))


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(test[['height', 'weight']]) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow',
             xy=(2, inertias[1]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1))
# plt.show()

# print('Centroids: '+ kmeans.cluster_centers_)

# print(kmeans.labels_)

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    
def plot_centroids(centroids, weights = None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color = circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker = 'x', s=2, linewidths=12,
                color = cross_color, zorder=11, alpha=1)
    

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids = True, show_xlabels = True, show_ylabels = True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(Z, extent = [mins[0], maxs[0], mins[1], maxs[1]], cmap = 'Pastel2')
    plt.contourf(Z, extent = [mins[0], maxs[0], mins[1], maxs[1]], color = "k", linewidths = 1)
    
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)
        
    if show_xlabels:
        plt.xlabel('height (inches)', fontsize=8)
    else:
        plt.tick_params(labelbottom = False)
    if show_ylabels:
        plt.ylabel('weight (lbs)', fontsize=8, rotation=0)
    else:
        plt.tick_params(labelleft = False)

plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, test[['height', 'weight']].values)


x = nhl_output[['Age','Pos','GP', 'G', 'A', 'P', 'TOI', 'PP', 'PPG']]
y = nhl_output[['height', 'weight']]

# output the correlation to a csv
nhl_output.corr().to_csv('nhl_output_corr.csv')

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(x) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(x)

# plt.figure(figsize=(8, 3.5))
# plt.plot(range(1, 10), inertias, "bo-")
# plt.xlabel("$k$", fontsize=14)
# plt.ylabel("Inertia", fontsize=14)
# plt.annotate('Elbow',
#              xy=(3, inertias[2]),
#              xytext=(0.55, 0.55),
#              textcoords='figure fraction',
#              fontsize=16,
#              arrowprops=dict(facecolor='black', shrink=0.1))
# plt.show()

#['+/-', 'TOI', 'ES', 'PP', 'SH', 'ESA', 'PPA', 'SHA', 'GWA', 'OTA', 'ESP', 
# 'PPP', 'SHP', 'G/60', 'A/60', 'P/60', 'ESG/60', 'ESA/60', 'ESP/60', 'PPG/60', 
# 'PPA/60', 'PPP/60', 'SHOTS', 'SH%', 'HITS', 'BS', 'FOW', 'FOL', 'FO%', 
# 'primary_number', 'birth_city', 'birth_state_province']

lr = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()

height_pred_x = nhl_output[['Pos','PIM', 'HITS', 'BS', 'Age']]
height_pred_y = nhl_output[['height']]

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(height_pred_x) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]
plot_inertias(inertias)

x_train, x_test, y_train, y_test = train_test_split(height_pred_x, height_pred_y, test_size=0.2, random_state=42)

kmeans = KMeans(n_clusters=2, random_state=42)
y_pred = kmeans.fit_predict(x_train, y_train)

# get each cluster 
cluster_0 = x_train[y_pred == 0]
cluster_1 = x_train[y_pred == 1]

# score each cluster
lr_reg = lr.fit(cluster_0, y_train[y_pred == 0])
print('Cluster 0 R^2: Height', lr.score(cluster_0, y_train[y_pred == 0]))
lr_reg = lr.fit(cluster_1, y_train[y_pred == 1])
print('Cluster 1 R^2: Height', lr.score(cluster_1, y_train[y_pred == 1]))

lr_reg = lr.fit(x_train, y_train)
dtr_reg = dtr.fit(x_train, y_train)
rfr_reg = rfr.fit(x_train, y_train.values.ravel())


print('Linear Regression R^2: Height ', lr_reg.score(x_test, y_test))
print('Decision Tree Regression R^2: Height ', dtr_reg.score(x_test, y_test))
print('Random Forest Regression R^2: Height ', rfr_reg.score(x_test, y_test))

weight_pred_x = nhl_output[['Age', 'Pos', 'GP', 'PIM', 'BS', 'current_age']]
weight_pred_y = nhl_output[['weight']]

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(weight_pred_x) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]
plot_inertias(inertias)

x_train, x_test, y_train, y_test = train_test_split(weight_pred_x, weight_pred_y, test_size=0.2, random_state=42)

kmeans = KMeans(n_clusters=2, random_state=42)
y_pred = kmeans.fit_predict(x_train, y_train)

lr_reg = lr.fit(cluster_0, y_train[y_pred == 0])
print('Cluster 0 R^2: Weight', lr.score(cluster_0, y_train[y_pred == 0]))
lr_reg = lr.fit(cluster_1, y_train[y_pred == 1])
print('Cluster 1 R^2: Weight', lr.score(cluster_1, y_train[y_pred == 1]))

# apply the linear regression
lm_reg = lr.fit(x_train, y_train)
dtr_reg = dtr.fit(x_train, y_train)
rfr_reg = rfr.fit(x_train, y_train.values.ravel())

print('Linear Regression R^2: Weight ', lm_reg.score(x_test, y_test))
print('Decision Tree Regression R^2: Weight ', dtr_reg.score(x_test, y_test))
print('Random Forest Regression R^2: Weight ', rfr_reg.score(x_test, y_test))