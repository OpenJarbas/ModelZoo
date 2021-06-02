import pickle
import random
from os.path import join, dirname

import nltk
from nltk import ClassifierBasedTagger
from nltk.chunk import tree2conlltags
from nltk.corpus import conll2000

from JarbasModelZoo.features import extract_iob_features
from JarbasModelZoo.nltk_chunkers import ClassifierChunkParser

TAGGER_META = {
    "corpus": "CONLL2000",
    "model_id": "nltk_conll2000_clf_chunk_tagger",
    "tagset": "",
    "lang": "en",
    "algo": "ClassifierBasedTagger",
    "backoff_taggers": ["UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk", "JarbasModelZoo"]
}

MODEL_META = {
    "corpus": "CONLL2000",
    "model_id": "nltk_conll2000_clf_chunker",
    "tagset": "",
    "lang": "en",
    "algo": "ClassifierChunkParser",
    "required_packages": ["nltk", "JarbasModelZoo"]
}
import json

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)

meta_path = join(META, TAGGER_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(TAGGER_META, f)

# initializing training and testing set
nltk.download('conll2000')
shuffled_conll_sents = list(conll2000.chunked_sents())

random.shuffle(shuffled_conll_sents)
train_sents = shuffled_conll_sents[:int(len(shuffled_conll_sents) * 0.9)]
test_sents = shuffled_conll_sents[int(len(shuffled_conll_sents) * 0.9 + 1):]

# Transform the trees in IOB annotated triples [(word, pos, chunk), ...]
chunked_sents = [tree2conlltags(sent) for sent in train_sents]
chunked_test = [tree2conlltags(sent) for sent in test_sents]


# make it compatible with the tagger interface [((word, pos), chunk), ...]
def triplets2tagged_pairs(iob_sent):
    return [((word, pos), chunk) for word, pos, chunk in iob_sent]


train_data = [triplets2tagged_pairs(sent) for sent in chunked_sents]
test_data = [triplets2tagged_pairs(sent) for sent in chunked_test]

# train the tagger
tagger = ClassifierBasedTagger(
    train=train_data,
    feature_detector=extract_iob_features)

a = tagger.evaluate(test_data)

print("Accuracy of clf chunk tagger: ", a)  # 0.9101376235704594

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "chunking", "nltk_conll2000_clf_chunk_tagger.pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)

# test the chunker
chunk = ClassifierChunkParser(path)

a = chunk.evaluate(test_sents)

print("Accuracy of clf chunker: ", a)
# ChunkParse score:
#     IOB Accuracy:  91.2%%
#     Precision:     84.8%%
#     Recall:        87.1%%
#     F-Measure:     85.9%%


path = join(dirname(dirname(dirname(__file__))),
            "models", "chunking", "nltk_conll2000_clf_chunker.pkl")

with open(path, "wb") as f:
    pickle.dump(chunk, f)
