import json
import pickle
import random
from os.path import join, dirname

import nltk
from nltk.corpus import brown
from nltk.tag.sequential import ClassifierBasedPOSTagger

MODEL_META = {
    "corpus": "brown",
    "lang": "en",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "model_id": "nltk_brown_clftagger",
    "tagset": "brown",
    "algo": "ClassifierBasedPOSTagger",
    "required_packages": ["nltk"]
}
# initializing training and testing set
nltk.download('brown')

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")

corpus = brown.tagged_sents()  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

tagger = ClassifierBasedPOSTagger(train=train_data)

a = tagger.evaluate(test_data)

MODEL_META["accuracy"] = a
print("Accuracy: ", a)
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)
