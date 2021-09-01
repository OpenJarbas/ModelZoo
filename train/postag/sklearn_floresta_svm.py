import json
import pickle
from os.path import join, dirname
from random import shuffle

import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from JarbasModelZoo.features import extract_word_features

MODEL_META = {
    "corpus": "floresta",
    "corpus_homepage": "http://www.linguateca.pt/Floresta",
    "tagset": "VISL (Portuguese)",
    "tagset_homepage": "https://visl.sdu.dk/visl/pt/symbolset-floresta.html",
    "lang": "pt",
    "model_id": "sklearn_floresta_svm_tagger",
    "algo": "sklearn.svm.LinearSVC",
    "required_packages": ["scikit-learn"]
}

nltk.download('floresta')


def clean_tag(t):
    if "+" in t: t = t.split("+")[1]
    if "|" in t: t = t.split("|")[1]
    if "#" in t: t = t.split("#")[0]
    t = t.lower()
    return t


floresta = [[(w, clean_tag(t)) for (w, t) in sent]
            for sent in nltk.corpus.floresta.tagged_sents()]
shuffle(floresta)

cutoff = int(len(floresta) * 0.9)
train_data = floresta[:cutoff]
test_data = floresta[cutoff:]


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        untagged = [w for w, t in tagged]
        for index in range(len(tagged)):
            X.append(extract_word_features(untagged, index))
            y.append(tagged[index][1])

    return X, y


X, y = transform_to_dataset(train_data)

# Use only the first 10K samples
# numpy.core._exceptions.MemoryError: Unable to allocate 3.21 TiB for an
# array...
X = X[:10000]
y = y[:10000]

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', LinearSVC())
])

clf.fit(X, y)

print('Training completed')

X_test, y_test = transform_to_dataset(test_data)
X_test = X_test[:10000]
y_test = y_test[:10000]

acc = clf.score(X_test, y_test)
print("Accuracy:", acc)

# Accuracy: 0.9304


MODEL_META["accuracy"] = acc

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")
with open(path, "wb") as f:
    pickle.dump(clf, f)
META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)


def pos_tag(sentence):
    tags = clf.predict(
        [extract_word_features(sentence, index)
         for index in range(len(sentence))])
    return zip(sentence, tags)
