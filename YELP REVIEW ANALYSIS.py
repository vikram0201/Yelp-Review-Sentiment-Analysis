
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords


# In[32]:


yelp = pd.read_csv("Path+\\yelp.csv")
yelp.head()


# In[33]:


yelp.info()


# In[34]:


yelp.describe()


# In[35]:


yelp['text length'] = yelp['text'].apply(len)
yelp.head()


# # data exploration

# In[36]:


g = sns.FacetGrid(data=yelp, col='stars')
g.map(plt.hist, 'text length', bins=50)


# Seems like overall, the distribution of text length is similar across all five ratings. However, the number of text reviews seems to be skewed a lot higher towards the 4-star and 5-star ratings.

# In[37]:


sns.boxplot(x='stars', y='text length', data=yelp)


# From the plot, looks like the 1-star and 2-star ratings have much longer text, but there are many outliers (which can be seen as points above the boxes). Because of this, maybe text length won’t be such a useful feature to consider after all.

# In[38]:


#Correlations between cool, useful, funny, and text length.
stars = yelp.groupby('stars').mean()
stars.corr()


# In[39]:


sns.heatmap(data=stars.corr(), annot=True)


# Looking at the map, funny is strongly correlated with useful, and useful seems strongly correlated with text length. We can also see a negative correlation between cool and the other three features

# Our task is to predict if a review is either bad or good, so let’s just grab reviews that are either 1 or 5 stars from the yelp dataframe. We can store the resulting reviews in a new dataframe called yelp_class.

# In[40]:


yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
yelp_class.shape
#We can see from .shape that yelp_class only has 4086 reviews, compared to the 10,000 reviews in the original dataset. 
#This is because we aren’t taking into account the reviews rated 2, 3, and 4 stars.


# In[41]:


# let’s create the X and y for our classification task. X will be the text column of yelp_class, and y will be the stars column
X = yelp_class['text']
y = yelp_class['stars']


# The classification algorithm will need some sort of feature vector in order to perform the classification task. The simplest way to convert a corpus to a vector format is the bag-of-words approach, where each unique word in a text will be represented by one number.

# In[42]:


import string
def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[43]:


#We can use Scikit-learn’s CountVectorizer to convert the text collection into a matrix of token counts. 
#Since there are many reviews, we can expect a lot of zero counts for the presence of a word in the collection. Because of this, Scikit-learn will output a sparse matrix.
#Let’s import CountVectorizer and fit an instance to our review text (stored in X), passing in our text_process function as the analyser.
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
len(bow_transformer.vocabulary_)


# In[44]:


#To illustrate how the vectoriser works, let’s try a random review and get its bag-of-word counts as a vector. Here’s the twenty-fifth review as plain-text:
review_25 = X[24]
review_25


# In[45]:


#Now let’s see our review represented as a vector:
bow_25 = bow_transformer.transform([review_25])
bow_25


# In[46]:


#This means that there are 24 unique words in the review (after removing stopwords). Two of them appear thrice, and the rest appear only once


# Now that we’ve seen how the vectorisation process works, we can transform our X dataframe into a sparse matrix. To do this, let’s use the .transform() method on our bag-of-words transformed object.

# In[47]:


X = bow_transformer.transform(X)


# In[48]:


print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)
# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
density


# Training data and test data
# As we have finished processing the review text in X, It’s time to split our X and y into a training and a test set using train_test_split from Scikit-learn. We will use 30% of the dataset for testing.

# In[49]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[50]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)


# In[51]:


preds = nb.predict(X_test)


# In[52]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))


# model is more biased towards positive reviews compared to negative ones.
# 
# In conclusion, although our model was a little biased towards positive reviews, it was fairly accurate with its predictions, achieving an accuracy of 92% on the test set.

# In[53]:


from sklearn.feature_extraction.text import  TfidfTransformer


# In[54]:


#Import Pipeline from sklearn.


from sklearn.pipeline import Pipeline


# In[55]:


#Now create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[56]:


X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[57]:


pipeline.fit(X_train,y_train)


# In[58]:


predictions = pipeline.predict(X_test)


# In[59]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# Looks like Tf-Idf actually made things worse! That is it for this project.
