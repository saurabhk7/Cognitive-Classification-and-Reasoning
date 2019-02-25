#!/usr/bin/env python
# coding: utf-8

# # Cognitive Intelligence and Knowledge Based Classification and Reasoning

# #### Importing Libraries, basic visualisation and preprocessing

# In[1]:


import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from kmodes.kmodes import KModes


# In[2]:


#variables
num_clusters = 3
url="https://raw.githubusercontent.com/mahakbansal/Cognitive-Classification-and-Reasoning/master/Feb25_students.csv"


# In[3]:


#read data from url into a dataframe
df = pd.read_csv(url)


# In[4]:


# Columns and their types
df.dtypes


# In[5]:


mapping_df=pd.read_csv("mapping.csv")
for feature_name in mapping_df.columns:
    if feature_name=='S No':
        continue
    max_value = mapping_df[feature_name].max()
    min_value = mapping_df[feature_name].min()
    mapping_df[feature_name] = (mapping_df[feature_name] - min_value) / (max_value - min_value)
mapping_df = mapping_df.drop(["S No"], axis=1)
print(mapping_df)


# In[6]:


w, h = 37, 9
weights = [[0 for x in range(w)] for y in range(h)]
for ind, column in enumerate(mapping_df.columns):
    weights[ind] = mapping_df[column]
print(weights)
#each row is an attribute, and each column is a question its mapped to, eg: weights[0][11] is value of aptitude of question 12 (0-based indexing)


# In[7]:


#Data preprocessing
df.drop_duplicates(subset ="Email Address", keep = 'last', inplace = True) 
df = df.loc[df['College Name?'] == 'Pune Institute of Computer Technology']

#dfemail: retains the email id for future mapping
dfemail = df.copy()

df = df.drop(['Timestamp','Email Address','Current Branch?', 'Current Year?'], axis = 1)
df = df.drop(['College Name?'], axis = 1)


# In[8]:


# Displays descriptive stats for all columns
df.describe()


# In[9]:


df.plot(kind='box')


# In[10]:


df.dtypes


# In[11]:


df.head()


# In[12]:


# dataframe to store only categorical attributes
dfcat = df.loc[:, df.columns != 'What is your Grade in College (GPA)?']


# In[13]:


# Categorical boolean mask
categorical_feature_mask = dfcat.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = dfcat.columns[categorical_feature_mask].tolist()


# In[14]:


# instantiate labelencoder object
le = LabelEncoder()


# In[15]:


#Display categorical columns in dataframe
dfcat[categorical_cols]


# In[16]:


# apply le on categorical feature columns
dfcat[categorical_cols] = dfcat[categorical_cols].apply(lambda col: le.fit_transform(col))


# In[17]:


#Display encoded Categorical values in dataframe
dfcat[categorical_cols].head()


# In[18]:


#Add a gpa column in categorical dataframe and convert to a categorical bin
dfcat.insert(loc=0, column='GPA', value=df['What is your Grade in College (GPA)?'])
bin = [-1,0,7,7.5,8,8.25,8.5,8.75,9,9.25,9.5,10]
category = pd.cut(dfcat['GPA'],bin)
dfcat.insert(loc=0, column='Binned GPA', value=category)


# In[19]:


dfcat.head(14)


# In[20]:


dfcat['Binned GPA'] = dfcat['Binned GPA'].astype('str') 
dfcat['Binned GPA'] = dfcat['Binned GPA'].map({'(-1.0, 0.0]': 0,'(0.0, 7.0]': 0, '(7.0, 7.5]': 1, '(7.5, 8.0]': 2, '(8.0, 8.25]': 3, '(8.25, 8.5]': 4, '(8.5, 8.75]': 5, '(8.75, 9.0]': 6, '(9.0, 9.25]': 7, '(9.25, 9.5]': 8, '(9.5, 10.0]': 9})
dfcat = dfcat.drop(['GPA'], axis = 1)
dfcat.isnull().values.any()


# ### Apply kmodes on the preprocessed data

# In[21]:


df_dummy = pd.get_dummies(dfcat)
df_dummy = df_dummy.iloc[1:,:]
dfemail = dfemail.iloc[1:,:]


# In[22]:


#dissimilaty matrix for weighted attributes calculation
attribute_index = 0
def ss_attribute(a, b, **_):
    cost = []
#     print("Working on attr: ",attribute_index)
    for i in range(0,len(a)):
        row_cost = 0
        for j in range(0,len(a[i])):
            if(a[i][j]!=b[j]):
                row_cost+=abs(a[i][j]-b[j])*weights[attribute_index][j]
        cost.append(row_cost)
        
    return np.array(cost)


# In[30]:


run_elbow = 0

if(run_elbow):
    
    num_init=20
    km = []
    clusters = []
    num_attribute_clusters = 3
    for i in range(0,len(weights)):
        xx=[]
        yy=[]
        for j in range(1,8):            
            attribute_index = i
            kmtemp = KModes(n_clusters=j, init='Huang', n_init=10, verbose=0, cat_dissim=ss_attribute)
            clusterstemp = kmtemp.fit_predict(df_dummy)
            print("Attribute: ",i," Clusters: ",j," n_init: ",10," Best cost: ",kmtemp.cost_)
            xx.append(j)
            yy.append(kmtemp.cost_)
        plt.plot(xx,yy,'go-',label='Cluster vs Cost')
        plt.show()
else:
    km = []
    clusters = []
    num_attribute_clusters = 3
    for i in range(0,len(weights)):
        attribute_index = i
        km.append(KModes(n_clusters=num_attribute_clusters, init='Huang', n_init=10, verbose=0, cat_dissim=ss_attribute))
        clusters.append(km[i].fit_predict(df_dummy))
        print("Attribute: ",i," Clusters: ",num_attribute_clusters," n_init: ",10," Best cost: ",km[i].cost_)
    for i in range(0,len(clusters)):
        cluster_name = 'cluster'+str(i)
        df_dummy[cluster_name] = clusters[i]


# In[31]:


df_dummy.head()


# In[32]:


# #dissimilaty matrix calculation
# def ss(a, b, **_):
#     cost = []
#     for i in range(0,len(a)):
#         row_cost = 0
#         for j in range(0,len(a[i])):
#             if(a[i][j]!=b[j]):
#                 row_cost+=abs(a[i][j]-b[j])
#         cost.append(row_cost)
        
#     return np.array(cost)


# #### Algorithm for finding the appropriate number of clusters

# In[33]:


# run_elbow = 0

# if(run_elbow):
#     xx=[]
#     yy=[]
#     num_init=20
#     for i in range(1,11):
#         km = KModes(n_clusters=i, init='Huang', n_init=num_init, verbose=0, cat_dissim=ss)
#         clusters = km.fit_predict(df_dummy)
#         print("Clusters: ",i," n_init: ",num_init," Best cost: ",km.cost_)
#         xx.append(i)
#         yy.append(km.cost_)
#     plt.plot(xx,yy,'go-',label='Cluster vs Cost')
#     plt.show()
# else:
#     km = KModes(n_clusters=num_clusters, init='Huang', n_init=40, verbose=0, cat_dissim=ss)
#     clusters = km.fit_predict(df_dummy)
#     print("Clusters: ",num_clusters," n_init: ",40," Best cost: ",km.cost_)
#     df_dummy['clusters'] = clusters


# In[34]:


# # Display results of kmodes
def display_emails(i):
    kmodes_labels = km[i].labels_
    print (kmodes_labels)


    arr=[]
    for i in range(0,num_clusters):
        arr.append([])
    for i in range(0, len(kmodes_labels)):
        arr[kmodes_labels[i]].append((dfemail.iloc[i])["Email Address"])
    for i in range(0,num_clusters):
        print("Cluster ",i," :",arr[i])
        print()


# In[35]:


def display_plots(i):
    # # Principal Component Analysis for dimentionality reduction
    pca = PCA(2)
    cluster_name = 'cluster'+str(i)
    # Turn the dummified df into two columns with PCA
    plot_columns = pca.fit_transform(df_dummy.iloc[:,:-9])

    # Plot based on the two dimensions, and shade by cluster label
    plt.scatter(x=plot_columns[:,1], y=plot_columns[:,0], c=df_dummy[cluster_name])
    plt.title(cluster_name)
    plt.show()


# In[36]:


for i in range(0,9):
    display_emails(i)   


# In[ ]:


for i in range(0,9):
    display_plots(i)


# In[ ]:





# In[ ]:


# # Heatmap for feature weightage visualisation

# categorical_cols.insert(0,'Binned GPA')
# plt.matshow(pca.components_,cmap='viridis')
# plt.yticks([0,1],['1st Comp','2nd Comp'],fontsize=10)
# plt.colorbar()
# plt.xticks(range(len(categorical_cols)),categorical_cols,rotation=65,ha='left')
# plt.show()


# In[ ]:





# In[ ]:


# # printing positive attributes of  each cluster 

# kmodescent = km.cluster_centroids_
# shape = kmodescent.shape
# # For each cluster mode (a vector of "1" and "0")
# # find and print the column headings where "1" appears.
# # If no "1" appears, assign to "no-skills" cluster.
# print (shape[0])
# for i in range(shape[0]):
#     if sum(kmodescent[i,:]) == 0:
#         print("\ncluster " + str(i) + ": ")
#         print("no-skills cluster")
#     else:
#         print("\ncluster " + str(i) + ": ")
#         cent = kmodescent[i,:]
#         for j in df_dummy.columns[np.nonzero(cent)]:
#             print(j)

