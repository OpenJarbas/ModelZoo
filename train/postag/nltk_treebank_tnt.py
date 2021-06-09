import json
import pickle
import random
from os.path import join, dirname

import nltk
from nltk.corpus import treebank
from nltk.tag import DefaultTagger
from nltk.tag import tnt

MODEL_META = {
    "corpus": "treebank",
    "lang": "en",
    "model_id": "nltk_treebank_tnt_tagger",
    "tagset": "Penn Treebank",
    "algo": "TnT",
    "backoff_taggers": ["DefaultTagger"],
    "required_packages": ["nltk"]
}

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")

# initializing training and testing set
nltk.download('treebank')

corpus = treebank.tagged_sents()  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

# initializing tagger
tnt_tagging = tnt.TnT(unk=DefaultTagger('NN'),
                      Trained=True)

# training
tnt_tagging.train(train_data)

# evaluating
a = tnt_tagging.evaluate(test_data)
MODEL_META["accuracy"] = a
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)
print("Accuracy of TnT Tagging : ", a)  # 0.892467083962875

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(tnt_tagging, f)
