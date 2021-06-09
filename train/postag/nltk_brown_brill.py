import json
import pickle
import random
from os.path import join, dirname

import nltk
from nltk import AffixTagger
from nltk.corpus import brown
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

MODEL_META = {
    "corpus": "brown",
    "model_id": "nltk_brown_brill_tagger",
    "tagset": "brown",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}
TAGGER_META = {
    "corpus": "brown",
    "model_id": "nltk_brown_ngram_tagger",
    "tagset": "brown",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "algo": "TrigramTagger",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}



# initializing training and testing set
nltk.download('brown')

corpus = [_ for _ in brown.tagged_sents()]  # 57340
random.shuffle(corpus)
cuttof = int(len(corpus) * 0.9)
train_data = corpus[:cuttof]
test_data = corpus[cuttof:]

# create tagger
affix = AffixTagger(train_data)
uni = UnigramTagger(train_data, backoff=affix)
bi = BigramTagger(train_data, backoff=uni)
tagger = TrigramTagger(train_data, backoff=bi)

TAGGER_META["accuracy"] = tagger.evaluate(test_data)
print("Accuracy of ngram tagger : ", TAGGER_META["accuracy"])  # 0.9224974329959557

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", TAGGER_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)
META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, TAGGER_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(TAGGER_META, f)


tagger = nltk.BrillTaggerTrainer(tagger, nltk.brill.fntbl37())
tagger = tagger.train(train_data, max_rules=100)

MODEL_META["accuracy"] = tagger.evaluate(test_data)
print("Accuracy of Brill tagger : ", MODEL_META["accuracy"])  # 0.9353010205150772

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")
with open(path, "wb") as f:
    pickle.dump(tagger, f)
META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)