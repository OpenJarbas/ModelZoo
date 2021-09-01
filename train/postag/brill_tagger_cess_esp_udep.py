import json
import pickle
from os.path import join, dirname
from random import shuffle

import nltk
from biblioteca.corpora import CessEspUniversal


MODEL_META = {
    "corpus": "cess_esp_udep",
    "corpus_homepage": "https://github.com/OpenJarbas/biblioteca/blob/master/corpora/create_cess.py",
    "lang": "es",
    "model_id": "nltk_cess_esp_udep_brill_tagger",
    "tagset": "Universal Dependencies",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["DefaultTagger", "AffixTagger", "UnigramTagger",
                        "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"],
    "train/test": "80/20"
}
NGRAM_META = {
    "corpus": "cess_esp_udep",
    "corpus_homepage": "https://github.com/OpenJarbas/biblioteca/blob/master/corpora/create_cess.py",
    "lang": "es",
    "model_id":  "nltk_cess_esp_udep_ngram_tagger",
    "tagset": "Universal Dependencies",
    "algo": "TrigramTagger",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"],
    "train/test": "80/20"
}


META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")


cess = CessEspUniversal()
corpus = list(cess.tagged_sentences())
shuffle(corpus)
cutoff = int(len(corpus) * 0.8)
train_data = corpus[:cutoff]
test_data = corpus[cutoff:]


def_tagger = nltk.DefaultTagger('NOUN')
affix_tagger = nltk.AffixTagger(
    train_data, backoff=def_tagger
)
unitagger = nltk.UnigramTagger(
    train_data, backoff=affix_tagger
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
print("Accuracy of ngram tagger : ", a)  # 0.9670816481919133
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", NGRAM_META["model_id"] + ".pkl")
with open(path, "wb") as f:
    pickle.dump(tagger, f)


tagger = nltk.BrillTaggerTrainer(tagger, nltk.brill.fntbl37())
tagger = tagger.train(train_data)

a = tagger.evaluate(test_data)

print("Accuracy of Brill tagger : ", a)  # 0.9730047760780784
MODEL_META["accuracy"] = a
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)

path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")
with open(path, "wb") as f:
    pickle.dump(tagger, f)
