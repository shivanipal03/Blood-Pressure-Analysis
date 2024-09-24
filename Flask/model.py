#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing data
df = pd.read_csv('patient_data.csv')
df.head()
df.rename(columns={'C' : 'Gender'},inplace=True)
df.to_csv('patient_data.csv', index=False)
df.info()
df.shape

#checking for null values
df.isnull().sum()

df['Stages'].unique()

df.replace({'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS', 
                      'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)'},
                      inplace = True)


# In[26]:


df['Stages'].unique()
df.replace({'No ': 'No', 
                      'Yes ': 'Yes', '121- 130': '121 - 130'},
                     inplace=True)





#converting categorical into numerical value
from sklearn.preprocessing import LabelEncoder

columns = ['Gender','Age','Severity','History','Patient','TakeMedication','BreathShortness',
           'VisualChanges','NoseBleeding','Whendiagnoused','Systolic','Diastolic','ControlledDiet']

label_encoder = LabelEncoder()
for col in columns:
    df[col] = label_encoder.fit_transform(df[col])
    

custom_mapping = {'NORMAL': 0, 'HYPERTENSION (Stage-1)': 1, 'HYPERTENSION (Stage-2)': 2, 'HYPERTENSIVE CRISIS': 3}

# Apply the mapping
df['Stages'] = df['Stages'].map(custom_mapping)

df.info()
# In[33]:


df['Stages'].unique()

'''
# In[14]:


df['Stages'].replace({'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS', 
                      'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)'}
                    )


# In[26]:


df['Stages'].unique()


# In[16]:


df['Stages'].replace({'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS', 
                      'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)'})


# In[34]:


df['Stages'].unique()


# In[35]:


df.describe()


# In[36]:

'''
gender_counts = df['Gender'].value_counts()
#Plotting the pie chart
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.0f%%', startangle=140)
plt.title('Gender Distribution')
plt.axis('equal')
plt.show()


# In[37]:


frequency = df['Stages'].value_counts()

frequency.plot(kind='bar')
plt.figure(figsize=(6,6))
plt.xlabel('Stages')
plt.ylabel('Frequency')
plt.title('Count of stages')
plt.show()


# In[38]:


sns.countplot(x='TakeMedication', hue='Severity', data=df)
plt.title('Count plot of TakeMedication by Severity')
plt.show()


# In[39]:


sns.pairplot(df)


# In[40]:


sns.pairplot(df[['Age', 'Systolic', 'Diastolic']])
plt.show()





#splitting the data into X and Y
X = df.drop('Stages' , axis = 1)
X
X.to_csv('patient_data_d.csv', index=False)


# In[42]:


Y = df['Stages']
Y


# In[43]:


#splitting into training and testing dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=11)


# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)

acc_lr = accuracy_score(y_test,y_pred)
c_lr = classification_report(y_test,y_pred)

print('Accuracy Score: ',acc_lr)
print(c_lr)



# In[45]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)

acc_rf = accuracy_score(y_test,y_pred)
c_rf = classification_report(y_test,y_pred)

print('Accuracy Score: ',acc_rf)
print(c_rf)


# In[46]:


from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)
y_pred = decision_tree_model.predict(x_test)

acc_dt = accuracy_score(y_test,y_pred)
c_dt = classification_report(y_test,y_pred)

print('Accuracy Score: ',acc_dt)
print(c_dt)


# In[47]:


from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(x_train, y_train)
y_pred = NB.predict(x_test)

acc_nb = accuracy_score(y_test,y_pred)
c_nb = classification_report(y_test,y_pred)

print('Accuracy Score: ',acc_nb)
print(c_nb)


# In[48]:


from sklearn.naive_bayes import MultinomialNB

mNB = MultinomialNB()
mNB.fit(x_train, y_train)
y_pred = mNB.predict(x_test)

acc_mnb = accuracy_score(y_test,y_pred)
c_mnb = classification_report(y_test,y_pred)

print('Accuracy Score: ',acc_mnb)
print(c_mnb)



prediction = random_forest.predict([[0,3,0,2,0,0,1,0,0,0,0,0,0]])

print(prediction[0])



if prediction[0] == 0:
    print("NORMAL")
elif prediction[0] == 1:
    print("HYPERTENSION (Stage-1)")
elif prediction[0] == 2:
    print("HYPERTENSION (Stage-2)")
else:
    print("HYPERTENSIVE CRISIS")




prediction = random_forest.predict([[0,3,0,2,0,0,1,0,0,0,0,0,0]])


# In[56]:


model = pd.DataFrame({'Model':['Linear Regression','Decision Tree Classifier','RandomForest Classifier','Gaussian Navie Bayes','Multinomial Navie Bayes'],
                      'Score':[acc_lr,acc_dt,acc_rf,acc_nb,acc_mnb],
                      })
model


import pickle
import warnings
pickle.dump(random_forest, open("model.pkl" , "wb"))



