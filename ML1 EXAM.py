#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd


# In[111]:


df=pd.read_csv("/Users/ks/Downloads/Paper1/bank.csv")


# In[112]:


df = pd.read_csv('/Users/ks/Downloads/Paper1/bank.csv',sep = ';')


# In[113]:


df


# In[114]:


df.describe()


# In[115]:


df.info()


# In[116]:


df['y'].value_counts()


# In[117]:


y1 = df['y']


# In[118]:


df.head()


# In[119]:


df.isnull().sum()


# In[120]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[121]:


num_var = ["age","duration","campaign", "pdays", "previous","emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

for i in num_var:
  sns.boxplot(x = y1,y = i,data = df)
  plt.show();


# In[122]:


plt.figure(figsize = (10,10))
sns.heatmap(df.corr())
plt.show()


# In[123]:


cat_var=[i for i in df.columns if i not in num_var]


# In[124]:


y1 = y1.values


# In[125]:


df1 = pd.get_dummies(df)


# In[126]:


df1.info()


# In[127]:


ax = sns.regplot(x="emp.var.rate", y="y_yes", order=1, data=df1, truncate=True)


# In[128]:


sns.regplot(x="nr.employed", y="y_yes", order=1, data=df1, truncate=True);


# In[129]:


sns.regplot(x="age", y="y_yes", order=3, data=df1, truncate=True);


# In[130]:


sns.regplot(x="duration", y="y_yes", order=1, data=df1, truncate=True);


# In[131]:


sns.regplot(x="contact_telephone", y="y_yes", order=1, data=df1, truncate=True)


# In[132]:


sns.regplot(x="month_sep", y="y_yes", order=1, data=df1, truncate=True)


# In[133]:


sns.regplot(x="cons.conf.idx", y="y_yes", order=1, data=df1, truncate=True)


# In[134]:


sns.regplot(x="poutcome_nonexistent", y="y_yes", order=1, data=df1, truncate=True)


# In[135]:


sns.regplot(x="education_unknown", y="y_yes", order=1, data=df1, truncate=True)


# In[136]:


sns.regplot(x="euribor3m", y="y_yes", order=1, data=df1, truncate=True)


# In[137]:


sns.regplot(x="age", y="y_yes", order=3, data=df1, truncate=True);


# In[138]:


sns.regplot(x="campaign", y="y_yes", order=1, data=df1, truncate=True)


# In[139]:


df1.loc[(df1['campaign'] >15) & (df1['y_yes']==1)]


# In[140]:


y_yes1 = df1['y_yes']
y_no1 = df1['y_no']


# In[141]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=350,
                                random_state=0)
forest.fit(df1,y1)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


# In[142]:


print("Feature ranking:")
for f in range(df1.shape[1]):
    print("%d. %s: %f " % (f + 1, df1.columns[indices[f]], importances[indices[f]]))


# In[143]:


plt.figure()
plt.figure(figsize=(15,10))
plt.title("Feature importances")
plt.bar(range(df1.shape[1]), importances[indices],color="b", yerr=std[indices], align="center")
plt.xticks(range(df1.shape[1]), indices)
plt.xlim([-1, df1.shape[1]])
plt.show();


# In[145]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[147]:


X_train, X_test, y_train, y_test = train_test_split(df1, y1, test_size=0.2, random_state=1, stratify=y1)


# In[149]:


from sklearn.metrics import accuracy_score

for i in range(10,300,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

    print(accuracy_score(knn.predict(X_test),y_test))


# In[150]:


t = pd.DataFrame()
t['y_yes'] = y_yes1
t['y_no'] = y_no1
t.head()


# In[151]:


print("For age upto 30")
print("Nth Call \t Efficiency")
for i in range(1,30):
    num = float(df1[(df1['age'] <= 30) & (df1['campaign']==i) & (t['y_yes']==1)].shape[0])
    den = 1+float(df1[(df1['age'] <= 30) & (df['campaign'] >= i)].shape[0])
    print (str((i))+" \t\t "+str(num/den))


# In[152]:


print("For age between 30-40")
print("Nth Call \t Efficiency")
for i in range(1,30):
    num = float(df1[(df1['age'] <= 40) & (df1['age'] > 30) & (df1['campaign']==i) & (t['y_yes']==1)].shape[0])
    den = 1+float(df[(df['age'] <= 40) & (df['age'] > 30) & (df['campaign'] >= i)].shape[0])
    print (str((i))+" \t\t "+str(num/den))


# In[154]:


total_calls = sum(df1['campaign'])
print(total_calls)


# In[156]:


extra_calls = sum(df1[df1['campaign']>6]['campaign']) - 6*df1[df1['campaign']>6].shape[0]
print(extra_calls)


# In[157]:


reduction=100*extra_calls/total_calls
print(reduction)


# In[158]:


total_sales=float(t[t['y_yes']==1].shape[0])
print(total_sales)


# In[159]:


less_costly_sales=float(df[(df['campaign'] <= 6) & (t['y_yes']==1)].shape[0])
print(less_costly_sales)


# In[160]:


sales_percent=100*less_costly_sales/total_sales
print(sales_percent)


# In[ ]:




