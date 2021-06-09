import pickle
from os.path import join, dirname
import json

from nltk.tag import ClassifierBasedTagger

from JarbasModelZoo.features import extract_iob_features
from JarbasModelZoo.nltk_chunkers import NamedEntityChunker, \
    conlltags2tree

MODEL_META = {
    "corpus": "CONLL2003",
    "lang": "en",
    "corpus_homepage": "https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003",
    "model_id": "nltk_clftagger_conll2003_NER",
    "tagset": "conll_iob",
    "algo": "ClassifierBasedTagger | NamedEntityChunker",
    "entitíes": ['ORG', 'LOC', 'MISC', 'PER'],
    "required_packages": ["nltk", "JarbasModelZoo"]
}

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)


# corpus handling
def read_connll2003(filename):
    with open(filename, 'r') as file_handle:
        file_content = file_handle.read()
        annotated_sentences = file_content.split('\n\n')
        for annotated_sentence in annotated_sentences:
            annotated_tokens = [seq for seq in
                                annotated_sentence.split('\n') if
                                seq]

            standard_form_tokens = []
            for idx, annotated_token in enumerate(
                    annotated_tokens):
                annotations = annotated_token.split(' ')
                word, tag, ner = annotations[0], annotations[1], \
                                 annotations[3]
                standard_form_tokens.append((word, tag, ner))

            # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
            # Because the classfier expects a tuple as input, first item input, second the class
            yield [((w, t), iob) for w, t, iob in
                   standard_form_tokens]


corpus_root = "/home/user/my_code/OpenJarbas/nlp_models/NER-datasets/CONLL2003/train.txt"
reader = read_connll2003(corpus_root)
training_samples = list(reader)

corpus_root = "/home/user/my_code/OpenJarbas/nlp_models/NER-datasets/CONLL2003/test.txt"
reader = read_connll2003(corpus_root)
test_samples = list(reader)


def train():
    # training
    tagger = ClassifierBasedTagger(
        train=training_samples,
        feature_detector=extract_iob_features)

    # save pickle
    path = join(dirname(dirname(dirname(__file__))),
                "models", "ner", MODEL_META["model_id"] + ".pkl")

    with open(path, "wb") as f:
        pickle.dump(tagger, f)


def test():
    path = join(dirname(dirname(dirname(__file__))),
                "models", "ner", MODEL_META["model_id"] + ".pkl")
    chunker = NamedEntityChunker(path)
    # accuracy test
    score = chunker.evaluate(
        [conlltags2tree([(w, t, iob) for (w, t), iob in iobs])
         for iobs in test_samples])
    print(score.accuracy())
    MODEL_META["accuracy"] = score.accuracy()
    with open(meta_path, "w") as f:
        json.dump(MODEL_META, f)


train()
test()  # 0.8743839197702824
