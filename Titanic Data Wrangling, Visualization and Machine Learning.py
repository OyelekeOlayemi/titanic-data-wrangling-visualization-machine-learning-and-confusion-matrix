#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# machine learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# Import dataset into notebook.

# In[2]:


data = pd.read_csv("titanic.csv")


# # Data Exploration

# In[3]:


data.head()


# Select the row (record) for the first passenger.

# In[4]:


data.iloc[0]


# Select rows (records) for the 1st to the 8th passenger.

# In[5]:


data.iloc[0:8]


# Select the columns Name and Age.

# In[6]:


data.loc[:,['Name','Age']]


# Select specific name in the dataset

# In[7]:


data.loc[data['Name'] == 'Braund, Mr. Owen Harris']


# This return no row, since no record with that name exist.

# In[8]:


data.loc[data['Name'] == 'Braund, Mr. Owen Harr']


# Check the numnber of female and male who survived and died.

# In[9]:


data.groupby(['Sex', 'Survived'])['PassengerId'].count()


# Check info of our columns.

# In[10]:


data.info()


# In[11]:


data.shape


# Drop all columns we do not need (PassengerId, Name, Ticket and Cabin)

# In[12]:


data.drop(data.columns[[0,3,8,10]], axis=1, inplace=True)


# In[13]:


data.head()


# Check for counts of missing values for each column.

# In[14]:


data[data.isnull().any(axis=1)].count()


# Drop all NaNs (null values).

# In[15]:


data_df=data.dropna()


# In[16]:


data_df.shape


# Our dataset has reduced to 712 rows, though we have cleared all the null values.

# In[17]:


data_df[data_df.isnull().any(axis=1)].count()


# Now our dataset has no missing value.

# Descriptive statistics of our features.

# In[18]:


data_df.describe()


# The minimum age is 0.4 which shows there are children under 1 year old onboard the titanic.

# # Visualization

# In[19]:


#We will check how Sex, Pclass and Embarked of Passengers affect their chances of survival.


Chart, items =plt.subplots(1,3,figsize=(25,5))

CGender = sns.barplot(x="Sex",y="Survived",data=data_df,ax=items[0])

CGender = CGender.set_ylabel("Survival Probability")

CClass = sns.barplot(x="Pclass",y="Survived",data=data_df,ax=items[1])

CClass = CClass.set_ylabel("Survival Probability")

CEmbarked = sns.barplot(x="Embarked",y="Survived",data=data_df,ax=items[2])

CEmbarked = CEmbarked.set_ylabel("Survival Probability")


# Females, Passengers in Class 1 and Passengers who embarked at 'C'  have higher chances of survival.

# In[20]:


fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(data_df['Age'], data_df['Survived'])

plt.xlabel('Age')
plt.ylabel('Survived')


# Age and Survival do not say much.

# In[21]:


fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(data_df['Fare'], data_df['Survived'])

plt.xlabel('Fare')
plt.ylabel('Survived')


# More people who paid fare between 0 and 100 survive or do not survive, there are some people who paid very high fare like 500 who definately survived.

# In[22]:


as_fig = sns.FacetGrid(data_df,hue='Sex',aspect=5)

as_fig.map(sns.kdeplot,'Age',shade=True)

oldest = data_df['Age'].max()

as_fig.set(xlim=(0,oldest))

as_fig.add_legend()


# More younger females than younger males are onboard while more older males are onboard than older females. 

# In[23]:


as_fig = sns.FacetGrid(data_df,hue='Pclass',aspect=5)

as_fig.map(sns.kdeplot,'Age',shade=True)

oldest = data_df['Age'].max()

as_fig.set(xlim=(0,oldest))

as_fig.add_legend()


# From the above graphs, we can infer that there are more number of passengers with a age group of 20 to 40 in all the three classes.

# In[24]:


f, ax = plt.subplots(2,2,figsize=(15,4))
vis1 = sns.distplot(data_df["Age"],bins=10, ax= ax[0][0])
vis2 = sns.distplot(data_df["Fare"],bins=10, ax=ax[0][1])
vis3 = sns.distplot(data_df["Survived"],bins=10, ax=ax[1][0])
vis4 = sns.distplot(data_df["Pclass"],bins=10, ax=ax[1][1])


# Most Passengers are between the age 20 - 40 years. Most passengers also paid between 0 to 100 dollars/pounds.

# In[25]:


vis5 = sns.boxplot(data = data_df, x = "Survived", y = "Age")
fig = vis5.get_figure()
fig.savefig("fig1.png")


# The boxplot shows we have extreme age between 60 - 80 years among those who survived or do not survived.  Also the median age of those who do not survived is more than those who survived.

# In[26]:


vis7 = sns.lmplot(data = data_df, x = "Fare", y = "Age",fit_reg=False, hue = "Survived",size = 6, aspect=1.5, scatter_kws = {'s':200}, )


# Only very few passengers paid above 100 pounds, majorly those that survived and whose age is lower than 60 years. Two particular passengers paid up to 500 pounds and are in their 40s.

# In[27]:


vis8= sns.lmplot(data = data_df, x = "Fare", y = "Age",fit_reg=False,                  hue = "Embarked",                 size = 6, aspect=1.5, scatter_kws = {'s':200},)


# Embarked Routes are:
# - Southampton, England 
# – Cherbourg, France 
# – Queenstown, Ireland
# 
# Passengers that paid the highest fares boarded the ship from Cherbourg in France.

# In[28]:


sns.countplot(y="Pclass", hue="Embarked", data=data_df);


# Embarked Routes are:
# - Southampton, England 
# – Cherbourg, France 
# – Queenstown, Ireland
# 
# Most Pclass 1, 2 and 3 Passengers boarded the ship at Southampton.
# Very few Pclass 1 and 2 passengers boarded the ship at Queenstown.

# In[29]:


vis9 = sns.swarmplot(x="Fare", y="Embarked", hue="Survived", data=data_df)
vis9.legend_.remove()
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.show()


# The swarm plot shows that most passengers that died boarded the ship at Southampton and paid between 0-300 pounds and the least that died boarded from Queenstown and paid less than 20 pounds.

# In[30]:


data_data_corr = data_df.corr()

data_data_corr


# There is a positive relationship between Fare and Survived. This means the higher the fare the higher the chances of survival.

# In[31]:


fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(data_data_corr, annot=True)


# The heatmap also shows the correlation between features. Survived and Fare have fairly high correlation.

# # Processing For Machine Learning Model

# Apply label encoding to change categorical string variable 'Sex' to represent '1' for Male and '0' for Female.

# In[32]:


from sklearn import preprocessing

label_encoding = preprocessing.LabelEncoder()
data_df['Sex'] = label_encoding.fit_transform(data_df['Sex'].astype(str))

data_df.head()


# Apply one hot-encoding to split 'Embarked' into dummies with seperate columns.

# In[33]:


data_df = pd.get_dummies(data_df, columns=['Embarked'])

data_df.head()


# In[34]:


data_df.shape


# We now have 10 columns due to the one hot-encoding we applied.

# Split our dataset into test and train set.

# In[35]:


from sklearn.model_selection import train_test_split

X = data_df.drop('Survived', axis=1)
Y = data_df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[36]:


x_train.shape, y_train.shape


# In[37]:


x_train.info()


# In[38]:


y_train


# The training set is a subset of the data set used to train a model.
# 
# x_train is the training data set.
# y_train is the set of labels to all the data in x_train.
# 
# The test set is a subset of the data set that you use to test your model after the model has gone through initial vetting by the validation set.
# 
# x_test is the test data set.
# y_test is the set of labels to all the data in x_test.
# 

# In[39]:


x_test.info()


# In[40]:


y_test


# # Logistic regression for classification
# 
# We would apply Logistic regression for classification. Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable.

# In[41]:


from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear').fit(x_train, y_train)


# Regularization helps to solve over fitting problem in machine learning. Overfitting model will be a very poor generalization of data. Regularization is therefore adding a penalty term to the objective function and control the model complexity using that penalty term. It can be used for many machine learning algorithms. 

# A regression model that uses L1 regularization technique is called Lasso Regression and model which uses L2 is called Ridge Regression.
# 
# The key difference between these two is the penalty term.
# 
# 1. Ridge regression adds “squared magnitude” of coefficient as penalty term.
# 
# 2. Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term.
# 
# The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.
# 

# For small values of C, regularization strength (λ ) increase which will create simple models which under-fit the data.
# 
# For big values of C, the power of regularization (λ ) will decrease which allowed the model to increase it's complexity, and therefore, over-fit the data.

# In[42]:


y_pred = logistic_model.predict(x_test)


# Predict the labels (Survived Column) of the  test data.

# In[43]:


y_pred.shape


# In[44]:


y_pred


# Lets check the y_pred against its actual label.

# In[45]:


pred_results = pd.DataFrame({'y_test': y_test,
                             'y_pred': y_pred})


# In[46]:


pred_results.head(10)


# Our predictions (y_pred) aligned with their true labels.

# In[47]:


print("Training set score: {:.3f}".format(logistic_model.score(x_train, y_train)))
print("Test set score: {:.3f}".format(logistic_model.score(x_test, y_test)))


# Our model performed well on both the training and test datasets. No obvious over-fitting.

# In[48]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[49]:


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("accuracy_score : ", acc)
print("precision_score : ", prec)
print("recall_score : ", recall)


# Our accuracy, precision and recall scores are impressive.

# In[50]:


from sklearn import metrics


# In[51]:


print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, y_pred)))


# ![image.png](attachment:image.png)

# # k-Nearest Neighbors
# 
# The k-NN algorithm is arguably the simplest machine learning algorithm. Building the model consists only of storing the training data set. To make a prediction for a new data point, the algorithm finds the closest data points in the training data set — its “nearest neighbors.”

# k-Nearest-Neighbors (k-NN) is a supervised machine learning model. Supervised learning is when a model learns from data that is already labeled. A supervised learning model takes in a set of input objects and output values. The model then trains on that data to learn how to map the inputs to the desired output so it can learn to make predictions on unseen data.
# 

# In[52]:


from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy = []

# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:

# build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)

# record training set accuracy
    training_accuracy.append(knn.score(x_train, y_train))

# record test set accuracy
    test_accuracy.append(knn.score(x_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')


# In[53]:


print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(x_test, y_test)))


# Our Model perform well on the training data but fair on the test data.

# In[54]:


knn = KNeighborsClassifier(n_neighbors=35)
knn.fit(x_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(x_test, y_test)))


# Increasing our n_neighbors/k_neighbors to 35 on the test and train datasets. The result is ok.

# In[55]:


#Predict with your Knn Model
y_predknn = knn.predict(x_test) 
print(metrics.accuracy_score(y_test, y_predknn))


# In[56]:


acc = accuracy_score(y_test, y_predknn)
prec = precision_score(y_test, y_predknn)
recall = recall_score(y_test, y_predknn)

print("accuracy_score : ", acc)
print("precision_score : ", prec)
print("recall_score : ", recall)


# Our accuracy, precision and recall scores are fairly ok.

# In[57]:


print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, y_predknn)))


# ![image.png](attachment:image.png)

# # What is Accuracy, Precision and Recall ?

# Theses metrics are used for evaluating classification models. They are contained in a matrix called Confusion Matrix. This matrix is a performance measurement technique for Machine learning classification.The Key concept of confusion matrix is that it calculates the number of correct &amp; incorrect predictions. It shows the path in which classification model is confused when it makes predictions.

# # Accuracy

# Classification accuracy is our starting point. It is the number of correct predictions made divided by the total number of predictions made, multiplied by 100 to turn it into a percentage.
# 
# Though we can achieve a high accuracy score, it might not mean the classifier is of overall good quality because there can be bias. i.e. Accuracy can be misleading. For example, in a problem where there is a large class imbalance, a model can predict the value of the majority class for all predictions and achieve a high classification accuracy.

# ![image.png](attachment:image.png)

# # Precision

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# What do you notice for the denominator? The denominator is actually the Total Predicted Positive! So the formula becomes.

# ![image.png](attachment:image.png)

# Precision talks about how precise is your true positives out of those predicted as positives i.e. how many of them are actual positive.
# Precision is a good measure to determine, when the costs of False Positive is high. 
# 
# E.g. For instance, email spam detection. In email spam detection, a false positive means that an email that is non-spam has been identified as spam (A case of False Positive). The email user might lose important emails if the precision is not high for the spam detection model. Here the cost of accepting False Positive is high or unbearable.

# # Recall

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# Recall actually calculates how many of the Actual Positives our model capture by labeling them as True Positive. Recall is used in selecting our best model when there is a high cost associated with False Negative.
# 
# 
# For instance, in fraud detection or sick patient detection. If a fraudulent transaction (Actual Positive) is predicted as non-fraudulent (Predicted Negative), the consequence can be very bad for the bank.
# 
# Similarly, in sick patient detection. If a sick patient (Actual Positive) goes through the test and predicted as not sick (Predicted Negative). The cost associated with False Negative will be extremely high if the sickness is contagious.

# # Specificity and Sensitivity 

# In medical diagnosis, test sensitivity is the ability of a test to correctly identify those with the disease (true positive rate). Thus, it is a measure of how well your classifier identifies positive cases.
# 
# Whereas test specificity is the ability of the test to correctly identify those without the disease (true negative rate).Thus, it is a measure of how well your classifier identifies negative cases. 

# Sensitivity describes how good the model is at predicting the positive class when the actual outcome is positive.

# ![image.png](attachment:image.png)

# Specificity is also called the false alarm rate as it summarizes how often a positive class is predicted when the actual outcome is negative.

# ![image.png](attachment:image.png)

# In[ ]:




