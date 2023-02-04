#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('Bank_ Data2009-14.csv')
data


# In[3]:


data.columns


# In[4]:


data.count()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.plot(kind='box',subplots=True,layout=(20,2),sharex=False,sharey=False,figsize=(20,40))
plt.show()


# In[8]:


data.hist(layout=(20,2),figsize=(20,40))
plt.show()


# In[9]:


data.drop(['Obs','account_id'],axis=1,inplace=True)

print(data.shape)


# In[10]:


data.info()


# In[11]:


data.hist(layout=(16,2),figsize=(20,50))
plt.show()


# In[12]:


data.describe()


# In[13]:


data.cov()


# In[14]:


data.kurtosis()


# In[15]:


data.skew()


# In[16]:


from scipy import stats
z1=stats.zscore(data['cardwdln'])
z2=stats.zscore(data['cardwdlt'])
z3=stats.zscore(data['bankcolt'])
z4=stats.zscore(data['bankrn'])
z5=stats.zscore(data['cardwdlnd'])
z6=stats.zscore(data['othcrnd'])
z7=stats.zscore(data['acardwdl'])
z8=stats.zscore(data['cashwdt'])
z9=stats.zscore(data['cardwdltd'])


# In[ ]:





# In[17]:


data.insert(0,"Z-Score_cardwdln", list(z1), True)
data.insert(0,"Z-Score_cardwdlt", list(z2), True) 
data.insert(0,"Z-Score_bankcolt", list(z3), True) 
data.insert(0,"Z-Score_bankrn", list(z4), True) 
data.insert(0,"Z-Score_cardwdlnd", list(z5), True) 
data.insert(0,"Z-Score_othcrnd", list(z6), True) 
data.insert(0,"Z-Score_acardwdl", list(z7), True) 
data.insert(0,"Z-Score_cashwdt", list(z8), True) 
data.insert(0,"Z-Score_cardwdltd", list(z9), True) 


# In[18]:


data


# In[19]:


data.loc[data['Z-Score_cardwdltd']>1.96,'cardwdltd']=np.nan
data.loc[data['Z-Score_cashwdt']<-1.96,'cashwdt']=np.nan

data.loc[data['Z-Score_acardwdl']>1.96,"acardwdl"]=np.nan
data.loc[data['Z-Score_acardwdl']<-1.96,"acardwdl"]=np.nan

data.loc[data['Z-Score_bankcolt']>1.96,'bankcolt']=np.nan
data.loc[data['Z-Score_bankcolt']<-1.96,'bankcolt']=np.nan

data.loc[data['Z-Score_bankrn']>1.96,'bankrn']=np.nan
data.loc[data['Z-Score_bankrn']<-1.96,'bankrn']=np.nan

data.loc[data['Z-Score_cardwdlnd']>1.96,'cardwdlnd']=np.nan
data.loc[data['Z-Score_cardwdlnd']<-1.96,'cardwdlnd']=np.nan

data.loc[data['Z-Score_othcrnd']>1.96,'othcrnd']=np.nan
data.loc[data['Z-Score_othcrnd']<-1.96,'othcrnd']=np.nan

data.loc[data['Z-Score_acardwdl']>1.96,'acardwdl']=np.nan
data.loc[data['Z-Score_acardwdl']<-1.96,'acardwdl']=np.nan

data.loc[data['Z-Score_cashwdt']>1.96,'cashwdt']=np.nan
data.loc[data['Z-Score_cashwdt']<-1.96,'cashwdt']=np.nan

data.loc[data['Z-Score_cardwdltd']>1.96,'cardwdltd']=np.nan
data.loc[data['Z-Score_cardwdltd']<-1.96,'cardwdltd']=np.nan


# In[20]:


data


# In[21]:


#now replace the value of nan with median values
data.info()


# In[22]:


data['cardwdltd']=data['cardwdltd'].fillna(data['cardwdltd'].median())
data['cashwdt']=data['cashwdt'].fillna(data['cashwdt'].median())
data['acardwdl']=data['acardwdl'].fillna(data['acardwdl'].median())
data['othcrnd']=data['othcrnd'].fillna(data['othcrnd'].median())
data['bankcolt']=data['bankcolt'].fillna(data['bankcolt'].median())
data['bankrn']=data['bankrn'].fillna(data['bankrn'].median())
data['cardwdlnd']=data['cardwdlnd'].fillna(data['cardwdlnd'].median())
data['cashwdt']=data['cashwdt'].fillna(data['cashwdt'].median())
data['cardwdltd']=data['cardwdltd'].fillna(data['cardwdltd'].median())


# In[23]:


data['cashwdt'].median()


# In[24]:


data=data.drop(['Z-Score_cardwdltd', 'Z-Score_cashwdt','Z-Score_acardwdl','Z-Score_othcrnd','Z-Score_cardwdlnd','Z-Score_bankrn','Z-Score_bankcolt','Z-Score_cardwdlt','Z-Score_cardwdln'], axis=1)
data.shape


# In[25]:


# from scipy import stats
# z1=stats.zscore(data['cardwdltd'])

# data.insert(0,"Z-score_cardwdltd",list(z1),True)

# data.loc[data['Z-score_cardwdltd']>1.96,"cardwdltd"]=np.nan

# data["cardwdltd"]=data['cardwdltd'].fillna(data['cardwdltd'].median())


# In[26]:


data.shape


# In[27]:


data.info()


# In[28]:


X=data.iloc[:,0:37]
y=data.iloc[:,-1]


# In[29]:


X


# In[30]:


y


# In[31]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[32]:


data.info()


# In[33]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

cat_vars=['sex','card','second','frequency','region']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(X[var], prefix=var)
    data1=X.join(cat_list)
    X=data1
cat_vars=['sex','card','second','frequency','region']
data_vars=X.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


# In[34]:


# cat_vars=['sex','card','second','frequency','region']
# for var in cat_vars:
#     cat_list='var'+'_'+var
#     cat_list=pd.get_dummies(X[var],prefix=var)
#     data1=X.join(cat_list)
#     X=data1
# cat_vars=['sex','card','second','frequency','region']
# data_vars=X.columns.values.tolist()
# to_keep=[i for i in data_vars if i not in cat_vars]


# In[35]:


X


# In[36]:


X['bankcolt']


# In[37]:


X.info()


# In[38]:


X['sex_M']


# In[39]:


X_data_final=X[to_keep]
X_data_final.columns.values


# In[40]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
y_data_final=labelencoder_y.fit_transform(y)
y_data_final


# In[ ]:





# In[41]:


from sklearn.metrics import confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[42]:



X_train,X_test,y_train,y_test=train_test_split(X_data_final,y_data_final,test_size=0.4,random_state=42)
#create the classifier
Logreg=LogisticRegression(max_iter=500)

#fit the classifier to the training data
Logreg.fit(X_train,y_train)

#predict the label of the test set
y_pred=Logreg.predict(X_test)

#compute and print the confusion matrices and the classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[43]:


#standardization
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix,classification_report


# In[44]:


X,y=make_classification(random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
pipe=make_pipeline(StandardScaler(),LogisticRegression())
pipe.fit(X_train,y_train)

pipe.score(X_test,y_test)


# In[45]:


#Normalization
from sklearn import preprocessing


# In[46]:


scaler=preprocessing.MinMaxScaler()
names=pd.DataFrame(X_data_final)
d1=scaler.fit_transform(names)
scaler_df=pd.DataFrame(d1)
scaler_df.head()


# In[47]:


scaler1=preprocessing.MinMaxScaler()
name1=pd.DataFrame(y_data_final)
d2=scaler.fit_transform(name1)
scaler_df2=pd.DataFrame(d2)
scaler_df2


# In[48]:


X_train,X_test,y_train,y_test=train_test_split(scaler_df,scaler_df2,random_state=42)

logreg=LogisticRegression(max_iter=500)

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))


# In[49]:


#PCA
from sklearn.preprocessing import StandardScaler
b=StandardScaler().fit_transform(X_data_final)
b


# In[66]:


from sklearn.decomposition import PCA
pca = PCA(n_components=19)
#principalComponents = pca.fit_transform(X_data_final)
# pp=pd.DataFrame(principalComponents)
# pp
# principalComponents
pca.fit(X_data_final
pca.explained_variance_ratio_
print(pca.singular_values_)


# In[51]:


x_train,x_test,y_train,y_test = train_test_split(pp,y_data_final)
x_train.head()
x_test.head()


# In[52]:


log=LogisticRegression(random_state=42)
log.fit(x_train,y_train)


# In[53]:


y_pred=log.predict(x_test)


# In[54]:


y_pred


# In[55]:


log.score(x_test,y_test)


# In[56]:



# Decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

predict = clf.predict(X_test)
print(classification_report(y_test,predict))
print("confusion matrix")
print(confusion_matrix(y_test,predict))


# In[ ]:




