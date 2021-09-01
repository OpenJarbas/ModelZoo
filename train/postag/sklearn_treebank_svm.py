import json
import pickle
import random
from os.path import join, dirname

import nltk
from nltk.corpus import treebank
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

from JarbasModelZoo.features import extract_word_features

MODEL_META = {
    "corpus": "treebank",
    "model_id": "sklearn_treebank_svm_tagger",
    "tagset": "Penn Treebank",
    "lang": "en",
    "algo": "sklearn.svm.LinearSVC",
    "required_packages": ["scikit-learn"]
}


# initializing training and testing set
nltk.download('treebank')

corpus = list(treebank.tagged_sents())  # 3914

random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]


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

# Accuracy: 0.9347


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
