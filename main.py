import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import re
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.head())


def process(data):
    # Clear Data
    def lower_text(txt):
        return txt.lower()

    def remote_no_text(txt):
        txt = re.sub(r"\d+", "", txt)
        return txt

    def remote_url(txt):
        txt = re.sub(r"http?\S+|www\.\S+", '', txt)
        return txt

    def cleanpunc(txt):
        cleaned = re.sub(r'[?|!|\'|"|#]', r'', txt)
        cleaned = re.sub(r'[.|,|)|(|)|\|/]', r' ', cleaned)
        return cleaned

    def clear(txt):
        txt = remote_no_text(txt)
        txt = remote_url(txt)
        txt = lower_text(txt)
        txt = cleanpunc(txt)
        return txt

    # Get feature

    data['keyword'] = data['keyword'].fillna('NaN')
    data['location'] = data['location'].fillna('NaN')
    le = preprocessing.LabelEncoder()
    le.fit(data['keyword'])
    data['keyword'] = le.transform(data['keyword'])
    data['text'] = data['text'].apply(clear)
    data = data.drop(['location', 'id'], axis=1)

    return data


def get_features(train, tesst):
    tfidf_vect = TfidfVectorizer(ngram_range=(1, 3), max_features=9000)
    tfidf_vect.fit(train['text'].tolist()+test['text'].tolist())
    xtrain_tfidf = tfidf_vect.transform(train['text'].tolist())
    xtest_tfidf = tfidf_vect.transform(test['text'].tolist())

    return xtrain_tfidf.toarray(), xtest_tfidf.toarray()


if __name__ == '__main__':
    data = process(train)
    data_test = process(test)
    feature_A_train, feature_A_test = get_features(data, data_test)
    feature_train = np.concatenate(
        (feature_A_train, train['keyword'].to_numpy().reshape(-1, 1)), axis=1)
    feature_test = np.concatenate(
        (feature_A_test, test['keyword'].to_numpy().reshape(-1, 1)), axis=1)

    # Evaluation


    print('OK')
    model_svm = LinearSVC()
    accuracies = cross_val_score(
        model_svm, X_train, y_train, scoring='accuracy', cv=3)
    for fold_idx, accuracy in enumerate(accuracies):
        print('Fold_idx :', fold_idx, 'Accuracy :', accuracies)


    # Make submission
    X_train, X_test, y_train, y_test = train_test_split(
        feature_train, train['target'].to_numpy(), test_size=0.1, random_state=2000)

    model_svm = LinearSVC()
    model_svm.fit(X_train, y_train)
    filename = 'model.sav'
    pickle.dump(model_svm, open(filename, 'wb'))
    print('DONE MODEL')
    loaded_model = pickle.load(open(filename, 'rb'))
    _pre = loaded_model.predict(X_test)

    print(accuracy_score(y_test,_pre))
    print('MAKE SUBMISSION')

    submiss = loaded_model.predict(feature_test)
    submission = pd.DataFrame({'id': test['id'], 'target':submiss})
    submission.to_csv ('submission.csv', index = None, header=True)

        
