import json
import pickle
import random
from os.path import join, dirname

import nltk
from nltk.corpus import brown
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

from JarbasModelZoo.features import extract_word_features

MODEL_META = {
    "corpus": "brown",
    "model_id": "sklearn_brown_dtree_tagger",
    "tagset": "brown",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "algo": "sklearn.tree.DecisionTreeClassifier",
    "required_packages": ["scikit-learn"]
}


# initializing training and testing set
nltk.download('brown')

corpus = [_ for _ in brown.tagged_sents()]  # 57340
random.shuffle(corpus)
cuttof = int(len(corpus) * 0.8)
train_data = corpus[:cuttof]
test_data = corpus[cuttof:]



def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            untagged = [w for w, t in tagged]
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
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])

clf.fit(X, y)

print('Training completed')

X_test, y_test = transform_to_dataset(test_data)
X_test = X_test[:10000]
y_test = y_test[:10000]

acc = clf.score(X_test, y_test)
print("Accuracy:", acc)

# Accuracy: 0.8289


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
