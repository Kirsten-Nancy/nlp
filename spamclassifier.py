import nltk 
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score 


messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                       names=["label", "message"])

stop_words = set(stopwords.words('english'))
# sentences = nltk.sent_tokenize(paragraph)

corpus = []
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

for i in range(len(messages)):
    sentence = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    sentence = sentence.lower()
    words = nltk.word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    processed_sent = ' '.join(lemmatized_words)
    corpus.append(processed_sent) 

# Creating the BOW -Bag Of Words Model
cv = CountVectorizer(max_features=5000)
# count matrix = X -> Independent features
X = cv.fit_transform(corpus)
count_array = X.toarray()
# print(count_array.shape) 
df = pd.DataFrame(data=count_array, columns=cv.get_feature_names_out())

# print(df.head(10))

# Dependent features -> ham/spam
y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

# Train test split -> Splits the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train model using Naive Bayes Classifier
model = MultinomialNB().fit(X_train, y_train)

y_pred = model.predict(X_test)

confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)