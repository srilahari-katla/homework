#%%% imports
import numpy as np
import pandas as pd
import cProfile

#%% read in the data
df = pd.read_excel("clinics.xls")
print(df.head())


#%% define the distance computation function
# Define a basic Haversine distance formula
def haversine(lat1, lon1, lat2, lon2):
    MILES = 3959
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    total_miles = MILES * c
    return total_miles

#%% define a function to compute distance, using a for loop
# Define a function to manually loop over all rows and return a series of distances
def haversine_looping(df):
    distance_list = []
    for i in range(0, len(df)):
        d = haversine(40.671, -73.985, df.iloc[i]['locLat'], df.iloc[i]['locLong'])
        distance_list.append(d)
    return distance_list
cProfile.run("df['distance'] = haversine_looping(df)")

#%%% vectorize code by using series and iterrows
# Haversine applied on rows via iteration%%timeit
haversine_series = []
for index, row in df.iterrows():
    haversine_series.append(haversine(40.671, -73.985, row['locLat'], row['locLong']))
#cProfile.run("df['distance'] = haversine_series")
df['distance'] = haversine_series

#%%% Optimize further 

# Timing apply on the Haversine function %%timeit
df['distance'] = df.apply(lambda row: haversine(40.671, -73.985, row['locLat'], row['locLong']), \
                          axis=1)

#%lprun -f haversine df.apply(lambda row: haversine(40.671, -73.985, row['latitude'], row['longitude']), axis=1)
#%%%%timeit
# Vectorized implementation of Haversine applied on Pandas series
df['distance'] = haversine(40.671, -73.985, df['locLat'], df['locLong'])

#%%
cProfile.run("df['distance'] = haversine(40.671, -73.985, df['locLat'], df['locLong'])")

#%%%%timeit
# Vectorized implementation of Haversine applied on NumPy arrays
df['distance'] = haversine(40.671, -73.985, df['locLat'].values, df['locLong'].values)

#%%
cProfile.run("df['distance'] = haversine(40.671, -73.985, df['locLat'].values, df['locLong'].values)")