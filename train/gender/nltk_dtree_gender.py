import json
import random
from os.path import join, dirname

from nltk.corpus import names
from nltk import DecisionTreeClassifier
from nltk.classify import accuracy

from JarbasModelZoo.features import extract_single_word_features

MODEL_META = {
    "corpus": "nltk names",
    "lang": "en",
    "corpus_homepage": "",
    "model_id": "nltk_dtree_gender_clf",
    "algo": "ClassifierBasedTagger",
    "required_packages": ["nltk", "JarbasModelZoo"]
}

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)

corpus = ([(name, 'male') for name in names.words('male.txt')] +
          [(name, 'female') for name in names.words('female.txt')])
corpus = [(extract_single_word_features(n), gender) for (n, gender) in corpus]

random.shuffle(corpus)

cutoff = int(len(corpus) * 0.8)
train_data = corpus[:cutoff]
test_data = corpus[cutoff:]


classifier = DecisionTreeClassifier.train(train_data)
acc = accuracy(classifier, test_data)
print(classifier.pseudocode(depth=4))
print(acc)  # 0.7042164883574575
