#!/usr/bin/env python
# coding: utf-8

# ### Analize dataset

# In[1]:


import pandas as pd
from plotly import express as px
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


# In[2]:


df = pd.read_csv(r"C:\Users\goret\Desktop\APC Cas Kaggle\sigma_cabs.csv")
df  


# In[3]:


columns_names = df.columns.values
columns_names


# In[4]:


x = df.iloc[:, 0:12].values
y = df.iloc[:, 12].values

print("Dimensionalitat de la BBDD:", df.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)


# In[5]:


print("Per veure estadístiques dels atributs numèrics de la BBDD:")
df.describe()


# In[ ]:





# In[6]:


# detect missing values in the given series object

df.isnull().sum()


# In[7]:


df["Type_of_Cab"].value_counts()
df["Type_of_Cab"] = df["Type_of_Cab"].fillna("F")


# In[8]:


df["Life_Style_Index"].value_counts()
df["Life_Style_Index"].describe()


# In[9]:


df["Confidence_Life_Style_Index"].value_counts()


# In[10]:


df = df.dropna(subset=["Life_Style_Index"])
df.isnull().sum()


# In[11]:


sum = 0
count = 0
avg = 0
for  i in df["Customer_Since_Months"].value_counts().iteritems():
    count += i[1]
    sum += i[0]*i[1]
avg = round(sum/count,0)
df["Customer_Since_Months"] = df["Customer_Since_Months"].fillna(avg)


# In[12]:


df["Var1"] = df["Var1"].fillna(25)


# In[13]:


df.isnull().sum()


# In[14]:


df


# In[15]:


# ara mateix tenim 0 missing values en la BBDD


# In[ ]:





# ### VISUALIZING DATA

# In[16]:


df.plot()


# In[17]:


# histogram
df.hist(bins = 20, figsize = (15, 10))


# In[18]:


corr = df.corr()
corr


# In[19]:


corr.style.background_gradient(cmap = 'coolwarm')


# In[20]:


# heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot = True, cmap = "YlGnBu")
plt.show()


# In[21]:


# Mirem la relació entre atributs utilitzant la funció pairplot
relacio = sns.pairplot(df)


# ### Normalize data

# In[22]:


# A one-hot encoding can be applied to the integer representation. 
# This is where the integer encoded variable is removed and a new binary variable is added 
# for each unique integer value.

def one_hot_encoding(column):
    df = pd.get_dummies(column,drop_first=True)
    return df

# La función get_dummies permite eliminar la primera de las columnas generadas 
# para cada característica codificada para evitar la denominada colinealidad 
# (que una de las características sea una combinación lineal de las otras), 
# lo que dificulta el correcto funcionamiento de los algoritmos.

Type_Of_Cab = one_hot_encoding(df["Type_of_Cab"])
Confidence_Life_Style_Index = one_hot_encoding(df["Confidence_Life_Style_Index"])
Destination_Type = one_hot_encoding(df["Destination_Type"])
Gender = one_hot_encoding(df["Gender"])
Type_Of_Cab = Type_Of_Cab.rename(columns={'B': 'Type_Of_Cab_B','C': 'Type_Of_Cab_C','D': 'Type_Of_Cab_D','E': 'Type_Of_Cab_E','F': 'Type_Of_Cab_F'})
Confidence_Life_Style_Index = Confidence_Life_Style_Index.rename(columns = {"B":"Confidence_Life_Style_Index_B","C":"Confidence_Life_Style_Index_C"})
Destination_Type = Destination_Type.rename(columns = {'B':'Destination_Type_B','C':'Destination_Type_C','D':'Destination_Type_D','E':'Destination_Type_E','F':'Destination_Type_F','G':'Destination_Type_G','H':'Destination_Type_H','I':'Destination_Type_I','J':'Destination_Type_J','K':'Destination_Type_K','L':'Destination_Type_L','M':'Destination_Type_M','N':'Destination_Type_N'})
print("Columns name changed Succesfully.")


# In[23]:


df_one_hot_encoded = pd.concat([df,Type_Of_Cab,Confidence_Life_Style_Index,Destination_Type,Gender], axis=1)
df_one_hot_encoded


# In[24]:


# normalitzem


# In[25]:


cols_to_drop = ["Trip_ID","Type_of_Cab","Confidence_Life_Style_Index","Destination_Type","Gender"]
df_final = df_one_hot_encoded.drop(cols_to_drop,axis = 1)


cols_to_be_normalized = ['Trip_Distance', 'Customer_Since_Months', 'Life_Style_Index','Customer_Rating', 'Cancellation_Last_1Month', 'Var1', 'Var2', 'Var3']
cols_not_to_be_normalized = ['Type_Of_Cab_B', 'Type_Of_Cab_C', 'Type_Of_Cab_D', 'Type_Of_Cab_E','Type_Of_Cab_F', 
                            'Confidence_Life_Style_Index_B','Confidence_Life_Style_Index_C', 'Destination_Type_B',
                            'Destination_Type_C', 'Destination_Type_D', 'Destination_Type_E',
                            'Destination_Type_F', 'Destination_Type_G', 'Destination_Type_H',
                            'Destination_Type_I', 'Destination_Type_J', 'Destination_Type_K',
                            'Destination_Type_L', 'Destination_Type_M', 'Destination_Type_N',
                            'Male','Surge_Pricing_Type']


# In[26]:


normalize = normalize(df_final[cols_to_be_normalized])
normalize = pd.DataFrame(normalize,columns=cols_to_be_normalized)
binarized = df_final[cols_not_to_be_normalized].reset_index()
df_final =  pd.concat([normalize,binarized], axis=1)

df_final


# #### Converting Categorical values to Numerical Values

# In[27]:


df.info()


# In[28]:


# Converting Categorical values to Numerical Values
cleanup_nums = {"Type_of_Cab": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G":7},
                "Confidence_Life_Style_Index": {"A": 1, "B": 2, "C": 3},
                "Destination_Type": {'A': 1, 'E': 5, 'B': 2, 'C': 3, 'G': 7, 'D': 4, 'F': 6, 'K': 11, 'L': 12, 'H': 8, 'I': 9, 'J': 10, 'M': 13,'N': 14},
                "Gender" :{'Male': 1, "Female": 2}}

df = df.replace(cleanup_nums)
df


# In[29]:


df.info()


# In[30]:


df


# In[31]:


df = df.drop(columns = ['Trip_ID'])


# In[32]:


df


# ### Model Building

# In[54]:


# funció per calcular el mse

import math

def mean_squeared_error(y1, y2):
    # comprovem que y1 i y2 tenen la mateixa mida
    assert(len(y1) == len(y2))
    mse = 0
    for i in range(len(y1)):
        mse += (y1[i] - y2[i])**2
    return mse / len(y1)

def mse(v1, v2):
    return ((v1 - v2)**2).mean()


# In[55]:


X = df_final.drop("Surge_Pricing_Type",axis = 1)
Y = df_final["Surge_Pricing_Type"] 

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=8,test_size=0.3)


# In[56]:


# Decision tree

model_tree = DecisionTreeClassifier(random_state=0)
model_tree.fit(X_train, Y_train)


# In[57]:


#%timeit 
Y_pred1 = model_tree.predict(X_test)

print(Y_pred1)


# In[58]:


print(confusion_matrix(Y_test, Y_pred1))


# In[59]:


print(accuracy_score(Y_test, Y_pred1))


# In[91]:


# Mostrem l'error (MSE)
MSE = mse(y, Y_pred1)

print("Mean squeared error: ", MSE)


# In[61]:


print(classification_report(Y_test, Y_pred1))


# In[65]:


# Logistic Regression

model_linear = LogisticRegression()
model_linear.fit(X_train, Y_train)


# In[66]:


#%timeit
Y_pred2 = model_linear.predict(X_test)

print(Y_pred2)


# In[67]:


print(confusion_matrix(Y_test, Y_pred2))


# In[68]:


print(accuracy_score(Y_test, Y_pred2))


# In[92]:


# Mostrem l'error (MSE)
MSE = mse(y, Y_pred2)

print("Mean squeared error: ", MSE)


# In[93]:


print(classification_report(Y_test, Y_pred2))


# In[94]:


# Random Forest Classifier

model_random = RandomForestClassifier(random_state=0)
model_random.fit(X_train, Y_train)


# In[95]:


#%timeit 
Y_pred3 = model_random.predict(X_test)

print(Y_pred3)


# In[96]:


print(confusion_matrix(Y_test, Y_pred3))


# In[97]:


print(accuracy_score(Y_test, Y_pred3))


# In[98]:


# Mostrem l'error (MSE)
MSE = mse(y, Y_pred3)

print("Mean squeared error: ", MSE)


# In[99]:


print(classification_report(Y_test, Y_pred3))


# In[100]:


# KNN

model_knn = KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
model_knn.fit(X_train, Y_train)


# In[101]:


#%timeit 
Y_pred4 = model_knn.predict(X_test)

print(Y_pred4)


# In[102]:


print(confusion_matrix(Y_test, Y_pred4))


# In[103]:


print(accuracy_score(Y_test, Y_pred4))


# In[104]:


# Mostrem l'error (MSE)
MSE = mse(y, Y_pred4)

print("Mean squeared error: ", MSE)


# In[105]:


print(classification_report(Y_test, Y_pred4))


# In[106]:


# Naive Bayes

model_nb = GaussianNB()
model_nb.fit(X_train,Y_train)


# In[107]:


#%timeit 
Y_pred5 = model_nb.predict(X_test)

print(Y_pred5)


# In[108]:


print(confusion_matrix(Y_test, Y_pred5))


# In[109]:


print(accuracy_score(Y_test, Y_pred5))


# In[110]:


# Mostrem l'error (MSE)
MSE = mse(y, Y_pred5)

print("Mean squeared error: ", MSE)


# In[111]:


print(classification_report(Y_test, Y_pred5))


# In[112]:


#XGB

model_XGB = XGBClassifier()
model_XGB.fit(X_train, Y_train)


# In[113]:


Y_pred6 = model_XGB.predict(X_test)

print(Y_pred6)


# In[114]:


print(confusion_matrix(Y_test, Y_pred6))


# In[115]:


print(accuracy_score(Y_test, Y_pred6))


# In[116]:


# Mostrem l'error (MSE)
MSE = mse(y, Y_pred6)

print("Mean squeared error: ", MSE)


# In[117]:


print(classification_report(Y_test, Y_pred6))


# In[118]:


result = {'Model Name':['Decision Tree', 'Logistic Regression', 'Random Forest', 'KNN', 'Naive Bayes', 'XGB'], 
          'Accuracy Score': [0.5739062827068568*100, 0.4323435303968183*100, 0.6907389133100087*100, 0.3490625280344487*100, 0.5223528004545318*100, 0.6977662151251458*100],
            'Mean squeared error': [4.832621312148501, 3.668678157636448, 4.86962303488317, 4.232764273888415, 4.271349295498694, 4.860928974481443]}


# In[119]:


res = pd.DataFrame.from_dict(result)
res


# #### Extreme Gradient Boosting (XGB) gives the Highest Accuracy Score

# ### PCA

# In[87]:


columns_names = df.columns.values
columns_names


# In[88]:


scaler=StandardScaler()
DatasetDefinitiu = df
df = DatasetDefinitiu.drop(['Surge_Pricing_Type'], axis=1) 
df = pd.DataFrame(DatasetDefinitiu, columns = ['Trip_Distance', 'Type_of_Cab', 'Customer_Since_Months',
       'Life_Style_Index', 'Confidence_Life_Style_Index',
       'Destination_Type', 'Customer_Rating', 'Cancellation_Last_1Month',
       'Var1', 'Var2', 'Var3', 'Gender'])
scaler.fit(df)
X_scaled=scaler.transform(df)

pca=PCA(n_components=df.shape[1]) 
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled) 
 
expl = pca.explained_variance_ratio_
#print('suma:',sum(expl[0:5]))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# In[89]:


from sklearn.pipeline import make_pipeline

data_new=df

pca_pipe = make_pipeline(StandardScaler(), PCA(n_components = 5))
pca_pipe.fit(data_new)

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']


print('----------------------------------------------------')
print('Percentatge de variança explicada per cada component')
print('----------------------------------------------------')
print(modelo_pca.explained_variance_ratio_)

pca_pipe = make_pipeline(StandardScaler(), PCA(n_components = 5))
pca_pipe.fit(data_new)

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
ax.bar(
    x      = np.arange(modelo_pca.n_components_) + 1,
    height = modelo_pca.explained_variance_ratio_
)

for x, y in zip(np.arange(len(data_new.columns)) + 1, modelo_pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Percentatge de variança explicada per cada component')
ax.set_xlabel('Component principal')
ax.set_ylabel('Per. variança explicada');
print(modelo_pca)


# ##### Amb això nomès podem dir que la primera component explica, aproximadament, un 20% de la variancia observada en les dades. Cap de les dos últimes superen el 1% de la variancia explicada.

# In[ ]:




