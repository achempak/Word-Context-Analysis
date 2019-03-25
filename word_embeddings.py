#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
import re
import string
import math
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from io import IOBase

#%%
#Read in raw data
raw_data = brown.words()

#%%
#Clean up data
data = [word for word in raw_data if not re.fullmatch('['+string.punctuation+']+',word)]
data = [word for word in data if not re.fullmatch('['+string.digits+']+',word)]
data = [word.lower() for word in data]
stop_words = set(stopwords.words('english'))
data = [word for word in data if not word in stop_words]

#%%
#Get the frequency distribution of all of the (cleaned up) words in corpus
fd = nltk.FreqDist(data)

#%%
#Get list of 1002 (Set C) and 5002 (Set V) most common words in corpus.
#Skip 2 most common words.
words = []
for a,b in fd.items():
    pair = [a,b]
    words.append(pair)
words.sort(key=lambda tup: tup[1], reverse=True)
V = words[2:5002]
C = words[2:1002]

#%%
#Produce matrix of counts, and two dictionaries, one to map
#words to indices, and one to map indices to words
V_set = []
C_set = []
V_index_set = []
C_index_set = []
for i in range(0,len(V)):
    V_set.append([V[i][0],i])
    V_index_set.append([i,V[i][0]])
for i in range(0,len(C)):
    C_set.append([C[i][0],i])
    C_index_set.append([i,V[i][0]])
V_dict = {key: value for (key,value) in V_set}
C_dict = {key: value for (key,value) in C_set}
V_index_dict = {key: value for (key,value) in V_index_set}
C_index_dict = {key: value for (key,value) in C_index_set}
counts = np.zeros(shape=(len(C),len(V)))
for i in range(2,len(data)-2):
    if data[i] in V_dict:
        for j in {i-2,i-1,i+1,i+2}:
            if data[j] in C_dict:
                counts[C_dict[data[j]], V_dict[data[i]]] += 1

#%%
#Get probabilities (p_cw and p_c)
p_cw = np.zeros(shape=(len(C),len(V)))
for i in range(0,5000):
    total_counts = 0
    for j in counts[:,i]:
        total_counts += j
    for j in range(0,1000):
        p_cw[j,i] = counts[j,i]/total_counts
p_c = np.zeros(1000)
c_total = 0
for item in C:
    c_total += item[1]
for i in range(0,len(C)):
    p_c[i] = (C[i][1]/c_total)**0.8

#%%
#Produce matrix of C-dimensional vectors for all w in V. Each row is vector.
vectors = np.zeros(shape=(len(V),len(C)))
delta = math.exp(-10)
for i in range(0,len(V)):
    for j in range(0,len(C)):
        arg = 0
        if p_cw[j,i]>delta:
            arg = math.log10(p_cw[j,i]/p_c[j])
        vectors[i,j] = max(0,arg)

#%%
#Do PCA on w-vectors to reduce to 100 dimensions
pca = PCA(n_components=100)
principal_150 = pca.fit_transform(vectors)
principal = principal_150[:,0:]
principal_norm = normalize(principal, axis=1, norm='l2')
print(pca.explained_variance_)

#%%
#k-means clustering using variant of cosine distance
#kmeans = KMeans(n_clusters=100,random_state=0,n_init=20).fit(principal_norm)
kmeans = SpectralClustering(n_clusters=100,random_state=0,assign_labels='kmeans',n_init=20,affinity='linear')

#%%
#Get labels for words and separate into two lists, one for labels and one for
#original word.
original_words = []
cluster_words = []
labels = kmeans.labels_
for i in range(0,len(V)):
    original_words.append(V[i][0])
    cluster_word_index = labels[i]
    cluster_words.append(V_index_dict[cluster_word_index])

#%%
#Sort clusters using dictionary that maps labels (centroids) to original data as
#keys and values, respectively
unique_labels = []
for label in labels:
    if not unique_labels.__contains__(label):
        unique_labels.append(label)
centroid_words = []
for label in unique_labels:
    centroid_words.append(V_index_dict[label])
d = {}
for label in centroid_words:
    d[label]=[]
for i in range(0,len(labels)):
    d[cluster_words[i]].append(original_words[i])

#%%
#Print clusters to different files for easy readability
for i in range(0,len(centroid_words)):
    file_name = 'clusters2\\'+'cluster'+str(i)+'.txt'
    with open(file_name,'w') as fp:
        for word in d[centroid_words[i]]:
            word_pair = centroid_words[i]+','+word+'\n'
            fp.write(word_pair)

#%%
#Nearest neighbor of a few selected words
nbrs = NearestNeighbors(n_neighbors=2,metric='cosine').fit(principal)
query_points=['communism','autumn','cigarette','pulmonary','mankind','africa',
'revolution','september','chemical','detergent','dictionary','storm','worship',
'chicago','information','society','vector','utopian','jail','government','effect',
'timber','death','president','order','therapist']
coordinates = []
for word in query_points:
    index = V_dict[word]
    coordinates.append(principal[index])
nbr1=nbrs.kneighbors(coordinates,2,return_distance=False)
result_points = []
for i,j in nbr1:
    result_points.append(V_index_dict[j])
nbr_df = pd.DataFrame()
nbr_df['Original Word'] = query_points
nbr_df['Nearest Neighbor'] = result_points
nbr_df.head(n=26)