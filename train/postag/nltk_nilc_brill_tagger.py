import json
import pickle
from os.path import dirname
from os.path import join
import random

import nltk
from biblioteca.corpora.external import NILC


MODEL_META = {
    "corpus": "NILC_taggers",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/nilc/tools/nilctaggers.html",
    "model_id": "nltk_nilc_brill_tagger",
    "tagset": "NILC",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/nilc/download/tagsetcompleto.doc",
    "lang": "pt-br",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}
NGRAM_META = {
    "corpus": "NILC_taggers",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/nilc/tools/nilctaggers.html",
    "model_id": "nltk_nilc_ngram_tagger",
    "tagset": "NILC",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/nilc/download/tagsetcompleto.doc",
    "lang": "pt-br",
    "algo": "TrigramTagger",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")


nilc = NILC()
data = [s for s in nilc.tagged_sentences()]
random.shuffle(data)
cutoff = int(len(data) * 0.9)
train_data = data[:cutoff]
test_data = data[cutoff:]

affix_tagger = nltk.AffixTagger(train_data)
unitagger = nltk.UnigramTagger(train_data, backoff=affix_tagger
                               )

bitagger = nltk.BigramTagger(
    train_data, backoff=unitagger
)
tagger = nltk.TrigramTagger(
    train_data, backoff=bitagger
)

a = tagger.evaluate(test_data)
NGRAM_META["accuracy"] = a
with open(join(META, NGRAM_META["model_id"] + ".json"), "w") as f:
    json.dump(NGRAM_META, f)
print("Accuracy of ngram tagger : ", a)  # 0.8661514978057623
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", NGRAM_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)

tagger = nltk.BrillTaggerTrainer(tagger, nltk.brill.fntbl37())
tagger = tagger.train(train_data)

a = tagger.evaluate(test_data)
MODEL_META["accuracy"] = a
with open(join(META, MODEL_META["model_id"] + ".json"), "w") as f:
    json.dump(MODEL_META, f)
print("Accuracy of Brill tagger : ", a)  # 0.877122686510208
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)
