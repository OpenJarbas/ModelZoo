import json
import pickle
import random
from os.path import join, dirname

import nltk
from nltk import AffixTagger
from nltk.corpus import treebank
from nltk.tag import DefaultTagger
from nltk.tag import RegexpTagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag import brill, brill_trainer

MODEL_META = {
    "corpus": "treebank",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "model_id": "nltk_treebank_brill_tagger",
    "tagset": "Penn Treebank",
    "lang": "en",
    "algo": "BrillTaggerTrainer",
    "backoff_taggers": ["DefaultTagger", "AffixTagger", "RegexpTagger",
                        "UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
# initializing training and testing set
nltk.download('treebank')

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")

corpus = treebank.tagged_sents()  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

# create tagger
patterns = [
    (r'^\d+$', 'CD'),
    (r'.*ing$', 'VBG'),  # gerunds, i.e. wondering
    (r'.*ment$', 'NN'),  # i.e. wonderment
    (r'.*ful$', 'JJ')  # i.e. wonderful
]
affix = AffixTagger(train_data, backoff=DefaultTagger('NN'))
rx = RegexpTagger(patterns, backoff=affix)
uni = UnigramTagger(train_data, backoff=rx)
bi = BigramTagger(train_data, backoff=uni)
tri = TrigramTagger(train_data, backoff=bi)


def train_brill_tagger(initial_tagger, train_sents, **kwargs):
    templates = [
        brill.Template(brill.Pos([-1])),
        brill.Template(brill.Pos([1])),
        brill.Template(brill.Pos([-2])),
        brill.Template(brill.Pos([2])),
        brill.Template(brill.Pos([-2, -1])),
        brill.Template(brill.Pos([1, 2])),
        brill.Template(brill.Pos([-3, -2, -1])),
        brill.Template(brill.Pos([1, 2, 3])),
        brill.Template(brill.Pos([-1]), brill.Pos([1])),
        brill.Template(brill.Word([-1])),
        brill.Template(brill.Word([1])),
        brill.Template(brill.Word([-2])),
        brill.Template(brill.Word([2])),
        brill.Template(brill.Word([-2, -1])),
        brill.Template(brill.Word([1, 2])),
        brill.Template(brill.Word([-3, -2, -1])),
        brill.Template(brill.Word([1, 2, 3])),
        brill.Template(brill.Word([-1]), brill.Word([1])),
    ]

    trainer = brill_trainer.BrillTaggerTrainer(initial_tagger, templates,
                                               deterministic=True)
    return trainer.train(train_sents, **kwargs)


brill_tagger = train_brill_tagger(tri, train_data)

a = brill_tagger.evaluate(test_data)
MODEL_META["accuracy"] = a
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)
print("Accuracy of Brill tagger : ", a)  # 0.9083099503561407

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(brill_tagger, f)
