import pandas as pd

#import dataframe

df = pd.read_table("smsspamcollection/SMSSpamCollection",
                   sep = '\t',
                   header=None,
                   names = ['label','sms_message'])

df.head()

#Mapping ham to 0 and spam to 1

df['label'] = df.label.map({'ham':0,'spam':1})

df.head()

#Training and Testing set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
#Applying bag of words processing to our dataset

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()

training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

#Implementing Naive bayes

from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)
predictions = naive_bayes.predict(testing_data)

#Evaluation

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

accuracy = accuracy_score(y_test,predictions)
precision = precision_score(y_test,predictions)
recall = recall_score(y_test,predictions)
f1 = f1_score(y_test,predictions)

print accuracy
print precision
print recall
print f1