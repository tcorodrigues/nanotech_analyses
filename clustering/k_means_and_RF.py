import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# read the data file
data = pd.read_csv('pubs.txt', sep='\t')
data2 = data.iloc[:-1,1:] #Remove names of countries and "world"


'''
Calculate differences between consecutive years (columns) as in:
difference_matrix = data2.diff(axis=1)
'''

df = data2['2001'] - data2['2000']
df2 = pd.DataFrame(df, columns=['diff1'])
df2['diff2'] = data2['2002'] - data2['2001']
df2['diff3'] = data2['2003'] - data2['2002']
df2['diff4'] = data2['2004'] - data2['2003']
df2['diff5'] = data2['2005'] - data2['2004']
df2['diff6'] = data2['2006'] - data2['2005']
df2['diff7'] = data2['2007'] - data2['2006']
df2['diff8'] = data2['2008'] - data2['2007']
df2['diff9'] = data2['2009'] - data2['2008']
df2['diff10'] = data2['2010'] - data2['2009']
df2['diff11'] = data2['2011'] - data2['2010']
df2['diff12'] = data2['2012'] - data2['2011']
df2['diff13'] = data2['2013'] - data2['2012']
df2['diff14'] = data2['2014'] - data2['2013']
df2['diff15'] = data2['2015'] - data2['2014']
df2['diff16'] = data2['2016'] - data2['2015']
df2['diff17'] = data2['2017'] - data2['2016']
df2['diff18'] = data2['2018'] - data2['2017']
df2['diff19'] = data2['2019'] - data2['2018']


# Normalizing data
data3 = MinMaxScaler().fit_transform(df2)

# Dim reduction
tsne = TSNE(n_components=2, random_state=1, perplexity=50, learning_rate=100).fit_transform(data3)

'''
# Plot data
plt.scatter(tsne[:,0], tsne[:,1])
plt.show()
plt.close()
'''


# Determine a reasonable number of clusters in tsne space
Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k).fit(tsne)
    Sum_of_squared_distances.append(km.inertia_)


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

km.labels_

for k in [2,3,4,5,6,7,8,9,10]:
	clusterer = KMeans(n_clusters=k)
	preds = clusterer.fit_predict(tsne)
	centers = clusterer.cluster_centers_
	score = silhouette_score(tsne, preds)
	print("For n_clusters = {}, silhouette score is {})".format(k, score))



# Cluster data and plot
kmeans = KMeans(n_clusters=4).fit(tsne)

labels_tsne_scale = kmeans.labels_

tsne_df_scale = pd.DataFrame(tsne, columns=['tsne1', 'tsne2'])
clusters_tsne_scale = pd.concat([tsne_df_scale, pd.DataFrame({'tsne_clusters':labels_tsne_scale})], axis=1)


plt.scatter(clusters_tsne_scale.iloc[:,0], clusters_tsne_scale.iloc[:,1], c=labels_tsne_scale, cmap='Set1', s=100, alpha=0.6)


plt.show()

kmeans.cluster_centers_




# Random forest classifier and Y-randomization

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import numpy as np


# repeated stratified 10-fold validation
scores = []
for i in range(10):
	kf = StratifiedKFold(10, shuffle=True, random_state=i)
	rf = RandomForestClassifier(n_estimators=100, random_state=1)
	cv = cross_validate(rf, data3, labels_tsne_scale, cv=kf, scoring='accuracy', n_jobs=5)
	scores += [np.sum(cv['test_score'])/10]

np.mean(scores) # 0.9076060606060606
np.std(scores) # 0.009726181798963642


       
# repeated adversarial control
scores_adv = []
for i in range(10):
	kf = StratifiedKFold(10, shuffle=True, random_state=i)
	rf = RandomForestClassifier(n_estimators=100, random_state=1)
	labels_tsne_scale2 = np.random.permutation(labels_tsne_scale)
	cv = cross_validate(rf, data3, labels_tsne_scale2, cv=kf, scoring='accuracy', n_jobs=5)
	scores_adv += [np.sum(cv['test_score'])/10]

np.mean(scores_adv) # 0.3995707070707071
np.std(scores_adv) # 0.018067785643892987









