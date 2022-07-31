#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv("loan_data.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data['credit.policy'].value_counts()


# In[6]:


data.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib.pyplot as plt
data.hist(bins=50,figsize=(20,15))


# In[9]:


from sklearn.model_selection import train_test_split
train_set, test_set =train_test_split(data ,test_size=.2 ,random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[10]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['credit.policy']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]


# In[11]:


strat_test_set['credit.policy'].value_counts()


# In[12]:


data=strat_train_set.copy()


# In[13]:


corr_matrix = data.corr()
corr_matrix['revol.util'].sort_values(ascending=False)


# In[14]:


data.plot(kind="scatter", x="dti", y="revol.util", alpha=1)


# In[15]:


data = strat_train_set.drop("not.fully.paid", axis=1)
data_label = strat_train_set["not.fully.paid"].copy()


# In[16]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(data)


# In[17]:


imputer.statistics_


# In[18]:


X = imputer.transform(data)


# In[19]:


data_tr = pd.DataFrame(X, columns=data.columns)


# In[20]:


data_tr.describe()


# In[21]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[22]:


data_num_tr = my_pipeline.fit_transform(data)


# In[23]:


data_num_tr.shape


# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
 #model = LinearRegression()
model = DecisionTreeRegressor()
#model = RandomForestRegressor()
model.fit(data_num_tr, data_label)


# In[25]:


some_data = data.iloc[:5]


# In[26]:


some_labels = data_label.iloc[:5]


# In[27]:


prepared_data = my_pipeline.transform(some_data)


# In[28]:


model.predict(prepared_data)


# In[29]:


list(some_labels)


# In[30]:


from sklearn.metrics import mean_squared_error
data_predictions = model.predict(data_num_tr)
mse = mean_squared_error(data_label, data_predictions)
rmse = np.sqrt(mse)


# In[31]:


rmse


# In[32]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, data_num_tr, data_label, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# rmse_scores

# In[33]:


rmse_scores


# In[34]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[35]:


print_scores(rmse_scores)


# In[36]:


from joblib import dump, load
dump(model, 'Dragon.joblib') 


# In[37]:


X_test = strat_test_set.drop("not.fully.paid", axis=1)
Y_test = strat_test_set["not.fully.paid"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))


# In[38]:


final_rmse


# In[39]:


prepared_data[0]


# In[43]:


from joblib import dump, load

model = load('Dragon.joblib') 
features = np.array([[1,3,0.1387,852.87,11.95761129,20.79,712,7260,188854,0,0,1,1]])
model.predict(features)


# In[ ]:




