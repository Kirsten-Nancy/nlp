import pandas as pd

import nltk 
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm

df = pd.read_csv('./fakenewsdetection/train.csv')

df.head()

X = df.drop('label', axis=1)
X.head()

# Dependent feature -> Label
y = df['label']
y.head()

# Dropping N/A values
df = df.dropna()

articles = df.copy()

articles.reset_index(inplace=True)
# After dropping None values, the indexes for NA values are fixed, this resets them

# print(articles['title'][0])

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

corpus = []
 
for i in range(len(articles)):
    sentence = re.sub('[^a-zA-Z]', ' ', articles['title'][i])
    sentence = sentence.lower()
    # Try replacing this with split to check if performance improves
    # words = nltk.word_tokenize(sentence)
    words = sentence.split()

    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    processed_sent = ' '.join(lemmatized_words)
    corpus.append(processed_sent)

# print(corpus[0])

# Try using the different types of vectorizers, tfidf
cv = CountVectorizer(max_features=8000, ngram_range=(1,3))
# Take combos of one, two, three words
X = cv.fit_transform(corpus).toarray()


y = articles['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


count_df = pd.DataFrame(X_train, columns=cv.get_feature_names_out())
count_df.head()


# # Use the multinomial classified
# classifier = MultinomialNB()

# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# acc_score = metrics.accuracy_score(y_test, y_pred)

# print('Naive pred', len(y_pred))
# print("accuracy: {:.1%}".format(acc_score))

# cm = metrics.confusion_matrix(y_test, y_pred)
# print(cm)

# Support Vector Machines
clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

svm_pred = clf.predict(X_test)

svm_acc_score = metrics.accuracy_score(y_test, svm_pred)
print("accuracy: {:.1%}".format(svm_acc_score))
print('here')
