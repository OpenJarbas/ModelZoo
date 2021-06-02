import json
import pickle
from os.path import join, dirname

import nltk
from nltk import AffixTagger
from nltk.corpus import brown
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

MODEL_META = {
    "corpus": "brown",
    "model_id": "nltk_brown_brill_tagger",
    "tagset": "brown",
    "lang": "en",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)

# initializing training and testing set
nltk.download('brown')

corpus = brown.tagged_sents()  # 57340
cuttof = int(len(corpus) * 0.9)
train_data = corpus[:cuttof]
test_data = corpus[cuttof:]

# create tagger
affix = AffixTagger(train_data)
uni = UnigramTagger(train_data, backoff=affix)
bi = BigramTagger(train_data, backoff=uni)
tagger = TrigramTagger(train_data, backoff=bi)

print("Accuracy of ngram tagger : ",
      tagger.evaluate(test_data))  # 0.9224974329959557

tagger = nltk.BrillTaggerTrainer(tagger, nltk.brill.fntbl37())
tagger = tagger.train(train_data, max_rules=100)

print("Accuracy of Brill tagger : ",
      tagger.evaluate(test_data))  # 0.9353010205150772

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)
