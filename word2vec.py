import nltk 
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score 
from gensim.models import Word2Vec


paragraph = """A very comprehensive, detailed and easy/fun-to-read introductory book 
on NLP that is ideal for an undergraduate (or beginner) level. Many fundamental concepts 
are explained with minimal mathematics with the focus on developing intuition on the how 
and why of things. This approach will appeal to many readers who are looking for a first book
on NLP and want to know the details of the underlying concepts. Plus, the easy-to-digest
explanations will help readers retain what they learn for the long term. NLP applications
have come very far from search engines and document sorting tools, understandably the author needs
to leave some details of the implementation. The good thing is – these have been stated clearly; therefore,
readers are well informed of such choices. This book misses talking about transformers that have revolutionised 
NLP in the last few years – something to include in a subsequent edition. Overall, this book helps build 
a good foundation that can be further developed-upon by diving into more advanced books."""

stop_words = set(stopwords.words('english'))
sentences = nltk.sent_tokenize(paragraph)

corpus = []
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

for i in range(len(sentences)):
    sentence = re.sub('[^a-zA-Z]', ' ', sentences[i])
    sentence = sentence.lower()
    words = nltk.word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    corpus.append(lemmatized_words)
    # break

model = Word2Vec(corpus, min_count=1)

words = model.wv.key_to_index

# Find word vectors
vector = model.wv['nlp']

similar = model.wv.most_similar('book')

print(similar) 