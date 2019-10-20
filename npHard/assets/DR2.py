#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px


# In[54]:


import numpy as np
import pandas as pd
import os
from PIL import Image

##def get_image_data(d):
##    #folder = os.listdir(img_fold)
##    #folder = [i for i in folder if i != 'target.txt']
##    #y = open(f'{img_fold}/target.txt').readlines()
##
###     y = list(map(int, y))
##    y=d['Y']
##    img_list = []
##    for i in d['X']:
##        #img = Image.open(f'{img_fold}/{i}')
##        img = i.resize((224, 224))
##        img = np.array(img)
##        img = img.astype('float') / 255
##        img = img.flatten()
##        img_list.append(img)
##    img_list = np.array(img_list)
##    df = pd.DataFrame(data = img_list[0:, 0:], index = [i for i in range(img_list.shape[0])], columns = [f'pixel{i}' for i in range(img_list.shape[1])])
##    b=df.columns
##    df = df.join(pd.DataFrame({'y': y}))
##    return df,b
##
##dff,col=get_image_data('bradley')


# In[37]:


mnist = load_digits()
X = mnist.data / 255.0
y = mnist.target
print(X.shape, y.shape)
# mnist.target.shape


# In[38]:


feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))


# In[51]:


dff.head()


# In[59]:


feat_cols=col
def pca_features(data,dim=2):
    pca = PCA(n_components=dim)
    pca_result = pca.fit_transform(data[feat_cols].values)
    print(pca_result.shape)
    data['one'] = pca_result[:,0]
    data['two'] = pca_result[:,1]
    if dim == 2:
        return data
    elif dim == 3:
        data['three'] = pca_result[:,2]
        return data

def tsne_features(data,dim=2):
    pca_50 = PCA(n_components=20)
    pca_result_50 = pca_50.fit_transform(data[feat_cols].values)
    tsne = TSNE(n_components=dim, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    data['one'] = tsne_pca_results[:,0]
    data['two'] = tsne_pca_results[:,1]
    if dim == 2:
        return data
    elif dim == 3:
        data['three'] = tsne_pca_results[:,2]
        return data
    


# In[56]:


pca_features(df.copy()).head()


# In[57]:


def plot3D(data,typ='PCA', y=None ,seed =42):
    np.random.seed(seed)
    dim=3
    rndperm = np.random.permutation(data.shape[0])
    print(rndperm)
    if typ == 'PCA' and y == None:
        dx = pca_features(data,dim)
        fig = px.scatter_3d(dx.loc[rndperm,:][:], x=dx.loc[rndperm,:]['one'], y=dx.loc[rndperm,:]['two'], z=dx.loc[rndperm,:]['three'],
              color=None)
        return fig
    elif typ == 'PCA' and y != None:
        dx = pca_features(data,dim)
        fig = px.scatter_3d(dx.loc[rndperm,:][:], x=dx.loc[rndperm,:]['one'], y=dx.loc[rndperm,:]['two'], z=dx.loc[rndperm,:]['three'],
              color=dx.loc[rndperm,:][y])
        return fig
    elif typ == 'TSNE' and y != None:
        dx = tsne_features(data,dim)
        fig = px.scatter_3d(dx.loc[rndperm,:][:], x=dx.loc[rndperm,:]['one'], y=dx.loc[rndperm,:]['two'], z=dx.loc[rndperm,:]['three'],
              color=dx.loc[rndperm,:][y])
        return fig
    elif typ == 'TSNE' and y == None:
        dx = tsne_features(data,dim)
        fig = px.scatter_3d(dx.loc[rndperm,:][:], x=dx.loc[rndperm,:]['one'], y=dx.loc[rndperm,:]['two'], z=dx.loc[rndperm,:]['three'],
              color=None)
        return fig
        
        


# In[58]:


plot3D(dff.copy(),'TSNE','y',34).show()


# In[64]:


plot3D(dff.copy(),'TSNE','y',34).show()


# In[60]:


def plot2D(data,typ='PCA', y=None ,seed =42):
    np.random.seed(seed)
    dim=2
    rndperm = np.random.permutation(data.shape[0])
    if typ == 'PCA' and y == None:
        dx = pca_features(data,dim)[rndperm,:]
        fig = px.scatter(dx.loc[rndperm,:], x='one', y='two',color=None)
        return fig
    elif typ == 'PCA' and y != None:
        dx = pca_features(data,dim)
        fig = px.scatter(dx.loc[rndperm,:], x='one', y='two',color=y)
        return fig
    elif typ == 'TSNE' and y != None:
        dx = tsne_features(data,dim)
        fig = px.scatter(dx.loc[rndperm,:],x='one', y='two',color=y)
        return fig
    elif typ == 'TSNE' and y == None:
        dx = tsne_features(data,dim)
        fig = px.scatter(dx.loc[rndperm,:], x='one', y='two',color=None)
        return fig


# In[62]:


plot2D(dff.copy(),'PCA','y',21).show()


# In[ ]:





# In[91]:



import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px

import os
from PIL import Image

def get_image_data(d,labels=False):
    folder = os.listdir(img_fold)
    if labels == True:
        #folder = [i for i in folder if i != 'target.txt']
        y = d['Y']
    img_list = []
    for i in d['X']:
        img = i.resize((224, 224))
        img = np.array(img)
        img = img.astype('float') / 255
        img = img.flatten()
        img_list.append(img)
    img_list = np.array(img_list)
    df = pd.DataFrame(data = img_list[0:, 0:], index = [i for i in range(img_list.shape[0])], columns = [f'pixel{i}' for i in range(img_list.shape[1])])
    b=df.columns
    if labels == True:
        df = df.join(pd.DataFrame({'y': y}))
    return df,b

#dff,col=get_image_data('bradley')


# In[37]:


def mains(typ,dim,seed,para='mnist',labels=False):
    np.random.seed(seed)
    if para=='mnist':
        
        mnist = load_digits()
        X = mnist.data / 255.0
        y = mnist.target
        #print(X.shape, y.shape)
        # mnist.target.shape


        # In[38]:
        feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
        df = pd.DataFrame(X,columns=feat_cols)
        df['y'] = y
        df['label'] = df['y'].apply(lambda i: str(i))
        rndperm = np.random.permutation(df.shape[0])
        if typ == 'PCA':
            pca = PCA(n_components=dim)
            pca_result = pca.fit_transform(df[feat_cols].values)
            #print(pca_result.shape)
            df['one'] = pca_result[:,0]
            df['two'] = pca_result[:,1]
            if dim == 3 :
                df['three'] = pca_result[:,2]
            
            return df.loc[rndperm,:],feat_cols
        elif typ == 'TSNE':
            pca_50 = PCA(n_components=20)
            pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
            tsne = TSNE(n_components=dim, verbose=0, perplexity=40, n_iter=300)
            tsne_pca_results = tsne.fit_transform(pca_result_50)
            df['one'] = tsne_pca_results[:,0]
            df['two'] = tsne_pca_results[:,1]
            if dim ==3:
                df['three'] = tsne_pca_results[:,2]
            return df.loc[rndperm,:],feat_cols
            
            
            
    else:
        labels = True
        df , feat_cols = get_image_data(dict1,labels)
        rndperm = np.random.permutation(df.shape[0])
        if typ == 'PCA':
            pca = PCA(n_components=dim)
            pca_result = pca.fit_transform(df[feat_cols].values)
            #print(pca_result.shape)
            df['one'] = pca_result[:,0]
            df['two'] = pca_result[:,1]
            if dim == 3 :
                df['three'] = pca_result[:,2]
            
            return df.loc[rndperm,:],feat_cols
        elif typ == 'TSNE':
            pca_50 = PCA(n_components=20)
            pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
            tsne = TSNE(n_components=dim, verbose=0, perplexity=40, n_iter=300)
            tsne_pca_results = tsne.fit_transform(pca_result_50)
            df['one'] = tsne_pca_results[:,0]
            df['two'] = tsne_pca_results[:,1]
            if dim ==3:
                df['three'] = tsne_pca_results[:,2]
            return df.loc[rndperm,:],feat_cols
        
        
        #print('Size of the dataframe: {}'.format(df.shape))
#a,b=mains('PCA',3,27,'mnist',True)

#print(b)
#a.head()


# In[92]:


#fig = px.scatter_3d(a, x='one', y='two', z='three',
              color='y')


# In[93]:


#fig.show()


# In[ ]:




