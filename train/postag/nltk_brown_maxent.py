import json
import pickle
import random
from os.path import join, dirname

import nltk
from nltk.classify import MaxentClassifier
from nltk.corpus import brown
from nltk.tag.sequential import ClassifierBasedPOSTagger

MODEL_META = {
    "corpus": "brown",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "model_id": "nltk_brown_maxent_tagger",
    "tagset": "brown",
    "algo": "ClassifierBasedPOSTagger",
    "classifier": "MaxentClassifier",
    "required_packages": ["nltk"]
}
# initializing training and testing set
nltk.download('brown')

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)
corpus = brown.tagged_sents()  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

tagger = ClassifierBasedPOSTagger(
    train=train_data, classifier_builder=MaxentClassifier.train)

a = tagger.evaluate(test_data)

MODEL_META["accuracy"] = a
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)

print("Maxent Accuracy : ", a)  # 0.9258363911072739

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)
