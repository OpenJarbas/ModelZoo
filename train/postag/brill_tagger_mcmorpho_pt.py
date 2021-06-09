import json
import pickle
from os.path import dirname
from os.path import join
from random import shuffle

import nltk

MODEL_META = {
    "corpus": "macmorpho",
    "model_id": "nltk_macmorpho_brill_tagger",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/macmorpho/",
    "tagset": "",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf",
    "lang": "pt",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["DefaultTagger", "AffixTagger", "UnigramTagger",
                        "RegexpTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")

nltk.download('mac_morpho')

path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")


def clean_tag(t, ):
    if "|" in t:
        t = t.split("|")[0]
    return t


dataset = [[(w, clean_tag(t)) for (w, t) in sent]
           for sent in nltk.corpus.mac_morpho.tagged_sents()]

shuffle(dataset)
print(dataset[0])

cutoff = int(len(dataset) * 0.9)
train_data = dataset[:cutoff]
test_data = dataset[cutoff:]

regex_patterns = [
    (r"^[nN][ao]s?$", "ADP"),
    (r"^[dD][ao]s?$", "ADP"),
    (r"^[pP]el[ao]s?$", "ADP"),
    (r"^[nN]est[ae]s?$", "ADP"),
    (r"^[nN]um$", "ADP"),
    (r"^[nN]ess[ae]s?$", "ADP"),
    (r"^[nN]aquel[ae]s?$", "ADP"),
    (r"^\xe0$", "ADP"),
]

def_tagger = nltk.DefaultTagger('N')
affix_tagger = nltk.AffixTagger(
    train_data, backoff=def_tagger
)
unitagger = nltk.UnigramTagger(
    train_data, backoff=affix_tagger
)
rx_tagger = nltk.RegexpTagger(
    regex_patterns, backoff=unitagger
)
bitagger = nltk.BigramTagger(
    train_data, backoff=rx_tagger
)
tagger = nltk.TrigramTagger(
    train_data, backoff=bitagger
)
a = tagger.evaluate(test_data)

print("Accuracy of ngram tagger : ", a)  # 0.9260627517381154

tagger = nltk.BrillTaggerTrainer(tagger, nltk.brill.fntbl37())
tagger = tagger.train(train_data, max_rules=100)

a = tagger.evaluate(test_data)
MODEL_META["accuracy"] = a
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)
print("Accuracy of Brill tagger : ", a)  # 0.9483812089870062

with open(path, "wb") as f:
    pickle.dump(tagger, f)
