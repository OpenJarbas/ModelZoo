import json
import pickle
from os.path import join, dirname

import nltk
from nltk.corpus import treebank
from nltk.tag import DefaultTagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

MODEL_META = {
    "corpus": "treebank",
    "lang": "en",
    "model_id": "nltk_treebank_ngram_tagger",
    "tagset": "Penn Treebank",
    "algo": "TrigramTagger",
    "backoff_taggers": ["DefaultTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}
# initializing training and testing set
nltk.download('treebank')

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")

corpus = treebank.tagged_sents()  # 3914
train_data = corpus[:3000]
test_data = corpus[3000:]


# create tagger
def backoff_tagger(train_sents, tagger_classes, backoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sents, backoff=backoff)
    return backoff


tagger = backoff_tagger(train_data,
                        [UnigramTagger, BigramTagger, TrigramTagger],
                        backoff=DefaultTagger('NN'))

a = tagger.evaluate(test_data)
MODEL_META["accuracy"] = a
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)
print("Accuracy of Ngram tagger : ", a)  # 0.8806388948845241

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)
