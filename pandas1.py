import pandas as pd
import numpy as np
california_housing_dataframe = pd.read_csv("mydata/california_housing_train.csv",
                                           sep=",")
#print(california_housing_dataframe.describe())
#print(california_housing_dataframe.head())
#print(california_housing_dataframe.hist('housing_median_age'))
#population = pd.Series([852469, 1015785, 485199])
#print(population/1000)
#print(population)
#print(population.apply(lambda val: val > 1000000))
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities['new'] = (cities['Area square miles'] > 50) \
                & cities['City name'].apply(lambda name: name.startswith('San'))
print(cities)
print(cities.reindex([2, 0, 1]))
print(cities.reindex(np.random.permutation(cities.index)))
print(cities.reindex(np.random.permutation(cities.index)))
print(cities.reindex([2, 0, 1,4,5]))
