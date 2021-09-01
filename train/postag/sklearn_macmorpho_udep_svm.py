import json
import pickle
import random
from os.path import join, dirname

import nltk
from random import shuffle
from string import punctuation
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from JarbasModelZoo.features import extract_word_features

MODEL_META = {
    "corpus": "macmorpho",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/lacioweb/",
    "tagset": "Universal Dependencies",
    "lang": "pt",
    "model_id": "sklearn_macmorpho_udep_svm_tagger",
    "algo": "sklearn.svm.LinearSVC",
    "required_packages": ["scikit-learn"]
}


nltk.download('mac_morpho')


def convert_to_universal_tag(t, reverse=True):
    tagdict = {
        'n': "NOUN",
        'num': "NUM",
        'v-fin': "VERB",
        'v-inf': "VERB",
        'v-ger': "VERB",
        'v-pcp': "VERB",
        'pron-det': "PRON",
        'pron-indp': "PRON",
        'pron-pers': "PRON",
        'art': "DET",
        'adv': "ADV",
        'conj-s': "CONJ",
        'conj-c': "CONJ",
        'conj-p': "CONJ",
        'adj': "ADJ",
        'ec': "PRT",
        'pp': "ADP",
        'prp': "ADP",
        'prop': "NOUN",
        'pro-ks-rel': "PRON",
        'proadj': "PRON",
        'prep': "ADP",
        'nprop': "NOUN",
        'vaux': "VERB",
        'propess': "PRON",
        'v': "VERB",
        'vp': "VERB",
        'in': "X",
        'prp-': "ADP",
        'adv-ks': "ADV",
        'dad': "NUM",
        'prosub': "PRON",
        'tel': "NUM",
        'ap': "NUM",
        'est': "NOUN",
        'cur': "X",
        'pcp': "VERB",
        'pro-ks': "PRON",
        'hor': "NUM",
        'pden': "ADV",
        'dat': "NUM",
        'kc': "ADP",
        'ks': "ADP",
        'adv-ks-rel': "ADV",
        'npro': "NOUN",
    }
    if t in ["N|AP", "N|DAD", "N|DAT", "N|HOR", "N|TEL"]:
        t = "NUM"
    if reverse:
        if "|" in t: t = t.split("|")[0]
    else:
        if "+" in t: t = t.split("+")[1]
        if "|" in t: t = t.split("|")[1]
        if "#" in t: t = t.split("#")[0]
    t = t.lower()
    return tagdict.get(t, "." if all(tt in punctuation for tt in t) else t)


dataset = [[(w, convert_to_universal_tag(t)) for (w, t) in sent]
           for sent in nltk.corpus.mac_morpho.tagged_sents()]

shuffle(dataset)

cutoff = int(len(dataset) * 0.8)
train_data = dataset[:cutoff]
test_data = dataset[cutoff:]



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

# Accuracy:  0.9204


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
