#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[28]:


import warnings
warnings.filterwarnings("ignore")
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


df=data = pd.read_csv('D:\patient_churn.csv')
df.columns


# In[30]:


#df.drop(['ID'],axis=1,inplace=True)


# In[31]:


df.isna().any()


# In[32]:


df['Death Year'].fillna(0, inplace=True)


# In[33]:


df.isna().any()


# In[34]:


data.columns=['Id','sex','Birth','Death','enrollment','visits','churn_A','churn_B','churn_C','churn_D','churn_E','churn_F']
data.head()


# In[35]:


conditions = [
                data['churn_A']==1,
              data['churn_B']==1,
              data['churn_C']==1,
              data['churn_D']==1,
              data['churn_E']==1,
              data['churn_F']==1
             ]
outputs = [
    'A','B','C','D','E','F'
]


# In[36]:


data['churn'] = np.select(conditions, outputs, 'Other')


# In[37]:


data.head()


# In[38]:


def model(X,y):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_recall_fscore_support as score
    #have split the data to 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #scaling the features
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test) 

    #trainign the radon forest classification model
    classify = RandomForestClassifier(n_estimators=20, random_state=0)  
    classify.fit(X_train, y_train)  
    y_pred = classify.predict(X_test)

    #creating confusion matrix to evaluate the metrics of the model
    cm = confusion_matrix(y_test, y_pred) 
    print(cm)

    #displying the accuracy
    print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
    

    #calculating the metrics
    precision, recall, fscore, support = score(y_test, y_pred)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    #print('support: {}'.format(support))


# In[39]:


#Model to classify churn for multiclass
X= data.iloc[:,1:5].values
y=data.iloc[:,12].values
model(X,y)


# In[40]:


#building the model for churnA
y=data.iloc[:,6]
model(X,y)


# In[41]:


#building the model for churnB
y=data.iloc[:,7]
model(X,y)


# In[42]:


#building the model for churnC
y=data.iloc[:,8]
model(X,y)


# In[43]:


#building the model for churnD
y=data.iloc[:,9]
model(X,y)


# In[44]:


#building the model for churnE
y=data.iloc[:,10]
model(X,y)


# In[45]:


#building the model for churnF
y=data.iloc[:,11]
model(X,y)


# In[ ]:





# In[ ]:




