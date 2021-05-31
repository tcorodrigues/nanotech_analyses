import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('cluster_2.txt', sep='\t')
data2 = data.iloc[:,2:] #Remove names of countries and "world"


# engineer features
data3 = data2.cumsum(axis=1)
data3['Sum'] = data2.sum(axis=1)


df = (data3['2001'])*100 / data3['Sum']
df2 = pd.DataFrame(df, columns=['diff1'])
df2['diff2'] = (data2['2002'])*100 / data3['Sum']
df2['diff3'] = (data2['2003'])*100 / data3['Sum']
df2['diff4'] = (data2['2004'])*100 / data3['Sum']
df2['diff5'] = (data2['2005'])*100 / data3['Sum']
df2['diff6'] = (data2['2006'])*100 / data3['Sum']
df2['diff7'] = (data2['2007'])*100 / data3['Sum']
df2['diff8'] = (data2['2008'])*100 / data3['Sum']
df2['diff9'] = (data2['2009'])*100 / data3['Sum']
df2['diff10'] = (data2['2010'])*100 / data3['Sum']
df2['diff11'] = (data2['2011'])*100 / data3['Sum']
df2['diff12'] = (data2['2012'])*100 / data3['Sum']
df2['diff13'] = (data2['2013'])*100 / data3['Sum']
df2['diff14'] = (data2['2014'])*100 / data3['Sum']
df2['diff15'] = (data2['2015'])*100 / data3['Sum']
df2['diff16'] = (data2['2016'])*100 / data3['Sum']
df2['diff17'] = (data2['2017'])*100 / data3['Sum']
df2['diff18'] = (data2['2018'])*100 / data3['Sum']
df2['diff19'] = (data2['2019'])*100 / data3['Sum']



# build isolation forest
data3 = df2

from sklearn.ensemble import IsolationForest
clf = IsolationForest(random_state=1).fit(data3)
preds = clf.predict(data3)



# save results and plot data
test_data = data.iloc[:-1]
test_data2 = pd.concat([data, pd.DataFrame({'iForest':preds})], axis=1)
test_data2.to_csv('results_iForest.txt', sep='\t')
test_data3 = pd.concat([df2, pd.DataFrame({'iForest':preds})], axis=1)

iran = test_data3.iloc[3,:]
japan = test_data3.iloc[6,:]
all_cluster2 = test_data3.drop(index=[3, 6])

columns = list(all_cluster2.keys())[:-1]
mean = []
std = []
for column in columns:
	av = all_cluster2[column].mean()
	rho = all_cluster2[column].std()
	mean.append(av)
	std.append(rho)


ci95 = [i * 1.96 for i in std]

upper_bound = list(np.array(mean) + np.array(ci95))
lower_bound = list(np.array(mean) - np.array(ci95))


x = np.arange(1,20)
plt.plot(x, mean, label='Cluster 2')
plt.fill_between(x, lower_bound, upper_bound, color='b', alpha=.1)
plt.plot(x, iran[:-1], label='Iran')
plt.plot(x, japan[:-1], label='Japan')

plt.xlim(1, 19)
plt.xticks(np.arange(min(x), max(x)+1, 2.0))
plt.ylabel('% Cumulative publication variation')
plt.xlabel('Year')
plt.legend()

plt.savefig('iForest.svg', sep='\t')
plt.show()








