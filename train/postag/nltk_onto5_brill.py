import json
import os
import pickle
import random
from os.path import join, dirname

import nltk
from nltk import AffixTagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

MODEL_META = {
    "corpus": "OntoNotes-5.0-NER-BIO",
    "lang": "en",
    "corpus_homepage": "https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO",
    "model_id": "nltk_onto5_brill_tagger",
    "tagset": "Penn Treebank",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)


# corpus handling
def read_ontonotes5(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".ner"):
                with open(os.path.join(root, filename), 'r') as file_handle:
                    file_content = file_handle.read()
                    annotated_sentences = file_content.split('\n\n')
                    if not annotated_sentences:
                        continue

                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in
                                            annotated_sentence.split('\n') if
                                            seq]
                        if not annotated_tokens:
                            continue
                        toks = []
                        for idx, annotated_token in enumerate(
                                annotated_tokens):
                            annotations = annotated_token.split('\t')
                            if not annotations:
                                continue
                            word, tag = annotations[0], annotations[1]
                            toks.append((word, tag))
                        yield toks


corpus_root = "/home/user/my_code/OpenJarbas/nlp_models/OntoNotes-5.0-NER-BIO"
reader = read_ontonotes5(corpus_root)

data = list(reader)
random.shuffle(data)
train_data = data[:int(len(data) * 0.9)]
test_data = data[int(len(data) * 0.9):]

# create tagger
affix = AffixTagger(train_data)
uni = UnigramTagger(train_data, backoff=affix)
bi = BigramTagger(train_data, backoff=uni)
tagger = TrigramTagger(train_data, backoff=bi)

print("Accuracy of ngram tagger : ",
      tagger.evaluate(test_data))  # 0.9126556850140619

tagger = nltk.BrillTaggerTrainer(tagger, nltk.brill.fntbl37())
tagger = tagger.train(train_data, max_rules=100)

print("Accuracy of Brill tagger : ",
      tagger.evaluate(test_data))  # 0.928649695021732

# save pickle
path = join(dirname(dirname(dirname(__file__))),
            "models", "postag", MODEL_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)
