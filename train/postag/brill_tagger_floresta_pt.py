import json
import pickle
from os.path import dirname
from os.path import join
from random import shuffle

import nltk

MODEL_META = {
    "corpus": "floresta",
    "corpus_homepage": "http://www.linguateca.pt/Floresta",
    "model_id": "nltk_floresta_brill_tagger",
    "tagset": "VISL (Portuguese)",
    "tagset_homepage": "https://visl.sdu.dk/visl/pt/symbolset-floresta.html",
    "lang": "pt",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}
NGRAM_META = {
    "corpus": "floresta",
    "corpus_homepage": "http://www.linguateca.pt/Floresta",
    "model_id": "nltk_floresta_ngram_tagger",
    "tagset": "VISL (Portuguese)",
    "tagset_homepage": "https://visl.sdu.dk/visl/pt/symbolset-floresta.html",
    "lang": "pt",
    "algo": "TrigramTagger",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")

nltk.download('floresta')


def clean_tag(t):
    if "+" in t: t = t.split("+")[1]
    if "|" in t: t = t.split("|")[1]
    if "#" in t: t = t.split("#")[0]
    t = t.lower()
    return t


floresta = [[(w, clean_tag(t)) for (w, t) in sent]
            for sent in nltk.corpus.floresta.tagged_sents()]
shuffle(floresta)

cutoff = int(len(floresta) * 0.9)
train_data = floresta[:cutoff]
test_data = floresta[cutoff:]

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
print("Accuracy of ngram tagger : ", a)  # 0.9272517853029256
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
print("Accuracy of Brill tagger : ", a)  # 0.9380327113568302
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)
