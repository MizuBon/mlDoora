"""
Created on Sun Nov 14 15:35:51 2021

@author: BonBon
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from sklearn.decomposition import PCA
import tensorflow_hub as hub
from pycaret.classification import * 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.metrics import accuracy_score
#from googletrans import Translator
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['font.serif'] = 'Ubuntu' 
plt.rcParams['font.monospace'] = 'Ubuntu Mono' 
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams['axes.labelweight'] = 'bold' 
plt.rcParams['axes.titlesize'] = 12 
plt.rcParams['xtick.labelsize'] = 12 
plt.rcParams['ytick.labelsize'] = 12 
plt.rcParams['legend.fontsize'] = 12 
plt.rcParams['figure.titlesize'] = 12 
plt.rcParams['image.cmap'] = 'jet' 
plt.rcParams['image.interpolation'] = 'none' 
plt.rcParams['figure.figsize'] = (10, 10
                                 ) 
plt.rcParams['axes.grid']=False
plt.rcParams['lines.linewidth'] = 2 
plt.rcParams['lines.markersize'] = 8
colors = ['xkcd:pale range', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
'xkcd:scarlet']
bbox_props = dict(boxstyle="round,pad=0.3", fc=colors[0], alpha=.5)
import pandas as pd
#import pycaret

#Loading the Dataset one moment
path = "https://raw.githubusercontent.com/MizuBon/mlDoora/main/safeDoors.csv"
#names = ['Program_Name', 'Code']
safeData = pd.read_csv(path)

path = "https://raw.githubusercontent.com/MizuBon/mlDoora/main/backdoors.csv"
#names = ['Program_Name', 'Code']
backData = pd.read_csv(path)

#The safe data and backdoor data CSVs get another column, called Target, that indicates whether or not it's safe or malicious
safeData['Target']=['Safe']*len(safeData)
backData['Target']=['Malicious']*len(backData)

#Combining the Datasets
data=backData.append(safeData).sample(frac=1).reset_index().drop(columns=['index'])
print(data)

#Allows us to see the distribution of data. Keep commented for most runs
#cat_tar=pd.get_dummies(data.Target)['Malicious']
#label_size = [cat_tar.sum(),len(cat_tar)-cat_tar.sum()]
#plt.pie(label_size,explode=[0.1,0.1],colors=['firebrick','navy'],startangle=90,shadow=True,labels=['Malicious','Safe'],autopct='%1.1f%%')

#Now let's try and train this baby
#Most of this code is taken from Fake News Machine Learning Project, although some modifications are made according to the dataset
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

#Splits up the data in training and testing sets
data_matrix = embed(data.Code.tolist())
train_data = data.loc[0:int(len(data)*0.8)]
test_data = data.loc[int(len(data)*0.8):len(data)]

pca = PCA(n_components=3)
pca_data = pca.fit(data_matrix[0:len(train_data)])
pca_train = pca.transform(data_matrix[0:len(train_data)])

pca_3_data = pd.DataFrame({'First Component':pca_train[:,0],'Second Component':pca_train[:,1],'Third Component':pca_train[:,2],'Target': train_data.Target})

plt.figure(figsize=(20,10))
plt.subplot(1,3,1)
sns.scatterplot(x='First Component', y = 'Second Component',hue='Target',data=pca_3_data,s=20)
plt.grid(True)
plt.subplot(1,3,2)
sns.scatterplot(x='First Component', y = 'Third Component',hue='Target',data=pca_3_data,s=20)
plt.grid(True)
plt.subplot(1,3,3)
sns.scatterplot(x='Second Component', y = 'Third Component',hue='Target',data=pca_3_data,s=20)
plt.grid(True)

pca_3_data
setup(data = pca_3_data, target='Target', fold_shuffle=True, session_id=2)
best_model = compare_models()
