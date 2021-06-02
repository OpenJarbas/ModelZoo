import pickle
import random
from os.path import join, dirname

import nltk
from nltk import TrigramTagger, BigramTagger, UnigramTagger
from nltk.chunk import tree2conlltags
from nltk.corpus import conll2000

from JarbasModelZoo.nltk_chunkers import PostagChunkParser

TAGGER_META = {
    "corpus": "CONLL2000",
    "model_id": "nltk_conll2000_postag_ngram_chunk_tagger",
    "tagset": "",
    "lang": "en",
    "algo": "TrigramTagger",
    "backoff_taggers": ["UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk", "JarbasModelZoo"]
}
MODEL_META = {
    "corpus": "CONLL2000",
    "model_id": "nltk_conll2000_postag_ngram_chunker",
    "tagset": "",
    "lang": "en",
    "algo": "PostagChunkParser",
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

# Extract only the (POS-TAG, IOB-CHUNK-TAG) pairs
train_data = [
    [(pos_tag, chunk_tag) for word, pos_tag, chunk_tag in tree2conlltags(sent)]
    for sent in train_sents]
test_data = [
    [(pos_tag, chunk_tag) for word, pos_tag, chunk_tag in tree2conlltags(sent)]
    for sent in test_sents]

# Train a NgramTagger
t1 = UnigramTagger(train_data)
t2 = BigramTagger(train_data, backoff=t1)
tagger = TrigramTagger(train_data, backoff=t2)
a = tagger.evaluate(test_data)

print("Accuracy of postag tagger: ", a)  # 0.8873818489203105

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "chunking",
            "nltk_conll2000_postag_ngram_chunk_tagger.pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)

chunk = PostagChunkParser(path)

a = chunk.evaluate(test_sents)

print("Accuracy of ngram postag chunker: ", a)
# ChunkParse score:
#     IOB Accuracy:  88.7%%
#     Precision:     80.4%%
#     Recall:        85.2%%
#     F-Measure:     82.7%%


path = join(dirname(dirname(dirname(__file__))),
            "models", "chunking", "nltk_conll2000_postag_ngram_chunker.pkl")

with open(path, "wb") as f:
    pickle.dump(chunk, f)
