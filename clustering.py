# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:57:23 2023

@author: divya
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def Expo(t, scale, growth):
    """
    Function to calculate exponential growth
    """
    f = (scale * np.exp(growth * (t-1961)))
    return f


def func(x, k, l, m):
    """
    Function for finding error ranges
    """
    k, x, l, m = 0, 0, 0, 0
    return k * np.exp(-(x-l)**2/m)


def err_ranges(x, func, param, sigma):
    """
    Function to find error ranges for fitted data
    x: x array of the data
    func: defined function above
    param: parameter found by fitted data
    sigma: sigma found by fitted data
    """
    import itertools as iter

    low = func(x, *param)
    up = low

    uplow = []
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        low = np.minimum(low, y)
        up = np.maximum(up, y)

    return low, up


def read_data(filename):
    """
    The below function reads data.
    """
    df = pd.read_excel(filename, skiprows=3)
    return df


def stat_data(df, col, value, yr, a):
    """
    The below function is used to filter data and returns a dataframe and 
    transpose of the dataframe.
    """
    df3 = df.groupby(col, group_keys=True)
    df3 = df3.get_group(value)
    df3 = df3.reset_index()
    df3.set_index('Indicator Name', inplace=True)
    df3 = df3.loc[:, yr]
    df3 = df3.transpose()
    df3 = df3.loc[:, a]
    df3 = df3.dropna(axis=1)
    return df3

# Reads csv file
Main_data = read_data("climate_change.xlsx")
start = 1961
end = 2015
year = [str(i) for i in range(start, end+1)]
# Data for fitting
Indicator = ['Population growth (annual %)',
             'Electricity production from oil sources (% of total)']
data = stat_data(Main_data, 'Country Name', 'Canada', year, Indicator)
data = data.rename_axis('Year').reset_index()
# Data for clustering
Indicator1 = ['Electricity production from oil sources (% of total)',
              'Electricity production from hydroelectric sources (% of total)',
              'Electricity production from coal sources (% of total)', 'Electricity production from renewable sources, excluding hydroelectric (kWh)',
              'Urban population (% of total population)']
data1 = stat_data(Main_data, 'Country Name', 'Australia', year, Indicator1)
print(data1.head())
data['Year'] = data['Year'].astype('int')
# Rename indicators to short
data1 = data1.rename(columns={
    'Electricity production from oil sources (% of total)': 'Electricity- oil source',
    'Electricity production from hydroelectric sources (% of total)': 'Electricity- hydroelectric sources',
    'Electricity production from coal sources (% of total)': 'Electricity- coal sources',
    'Electricity production from renewable sources, excluding hydroelectric (kWh)': 'Electricity- renewable sources',
    'Urban population (% of total population)': 'Urban population'})

# Plot fitting
plt.figure()
sns.scatterplot(data=data, x="Year",
                y="Population growth (annual %)", cmap="Accent")
plt.title('Scatter Plot between 1961-2015 before fitting')
plt.ylabel('Population growth (annual %)')
plt.xlabel('Year')
plt.xlim(1961, 2020)
plt.savefig("Scatter_fit.png")
plt.show()

popt, pcov = opt.curve_fit(
    Expo, data['Year'], data['Population growth (annual %)'], p0=[1000, 0.02])
data["Pop"] = Expo(data['Year'], *popt)
sigma = np.sqrt(np.diag(pcov))
low, up = err_ranges(data["Year"], Expo, popt, sigma)
#Plotting the fitted and real data by showing confidence range
plt.figure()
plt.title("Plot After Fitting")
plt.plot(data["Year"], data['Population growth (annual %)'], label="data")
plt.plot(data["Year"], data["Pop"], label="fit")
plt.fill_between(data["Year"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Population growth")
plt.savefig("Afterfitting.png")
plt.show()

#Predicting future values
low, up = err_ranges(2030, Expo, popt, sigma)
print("Population growth (annual %) in 2030 is ", low, "and", up)
low, up = err_ranges(2040, Expo, popt, sigma)
print("Population growth (annual %) in 2040 is ", low, "and", up)
def map_corr(df, size=6):
    """Function creates heatmap of correlation matrix for each pair of 
    columns in the dataframe.

    """

    import matplotlib.pyplot as plt  

    corr = df.corr()
    plt.figure(figsize=(size, size))
    plt.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.title("Australia's Heatmap".upper(), size = 12, fontweight = 'bold')
    plt.savefig('Heatmap.png', dpi = 300, bbox_inches = 'tight')
    
corr = data1.corr()
print(corr)
map_corr(data1)
plt.show()

# Plotting scatter matrix
pd.plotting.scatter_matrix(data1, figsize=(12, 12), s=5, alpha=0.8)
plt.tight_layout()
plt.title("Scatter Matrix")
plt.savefig("Scattermatrix.png")
plt.show()

plt.scatter(data1['Electricity- oil source'],
            data1['Electricity- hydroelectric sources'])
plt.title("Electricity production from oil source vs hydroelectric source")
plt.savefig("Scatterelectricity.png")

scaler = MinMaxScaler()
scaler.fit(data1[['Electricity- oil source']])
data1['Scaler_Oil'] = scaler.transform(
    data1['Electricity- oil source'].values.reshape(-1, 1))

scaler.fit(data1[['Electricity- hydroelectric sources']])
data1['Scaler_H'] = scaler.transform(
    data1['Electricity- hydroelectric sources'].values.reshape(-1, 1))
data_c = data1.loc[:, ['Scaler_Oil', 'Scaler_H']]


def n_cluster(data_frame):
  k_rng = range(1, 10)
  sse = []
  for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit_predict(data_frame)
    sse.append(km.inertia_)
  return k_rng, sse

#Using elbow method
plt.figure()
a, b = n_cluster(data_c)
plt.xlabel('k')
plt.ylabel('sum of squared error')
plt.title("Number of clusters")
plt.plot(a, b)
plt.savefig("Elbow.png")
plt.show()

km = KMeans(n_clusters=2) # finding best cluster point is 2
pred = km.fit_predict(data_c[['Scaler_Oil', 'Scaler_H']])
data_c['cluster'] = pred

# plotting clustering
centers = km.cluster_centers_
dc1 = data_c[data_c.cluster == 0]
dc2 = data_c[data_c.cluster == 1]
plt.scatter(dc1['Scaler_Oil'], dc1['Scaler_H'], color='pink')
plt.scatter(dc2['Scaler_Oil'], dc2['Scaler_H'], color='blue')
plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='*', color='red')
plt.title("Cluster plot with 2 clusters")
plt.xlabel("Electricity production from oil sources")
plt.ylabel("Electricity production from hydroelectric sources")
plt.savefig("Scatterwithcenter.png")


