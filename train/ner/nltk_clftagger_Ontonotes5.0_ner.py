import json
import pickle
from os.path import join, dirname

from nltk.tag import ClassifierBasedTagger

from JarbasModelZoo.features import extract_iob_features
from JarbasModelZoo.nltk_chunkers import NamedEntityChunker, \
    conlltags2tree

MODEL_META = {
    "corpus": "OntoNotes-5.0-NER-BIO",
    "lang": "en",
    "corpus_homepage": "https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO",
    "model_id": "nltk_clftagger_onto5_NER",
    "tagset": "conll_iob",
    "algo": "ClassifierBasedTagger | NamedEntityChunker",
    "entities": ['PERSON', 'FAC', 'EVENT', 'GPE', 'TIME',
                 'LAW', 'PRODUCT', 'ORDINAL', 'ORG', 'QUANTITY',
                 'CARDINAL', 'LOC', 'DATE', 'WORK_OF_ART', 'NORP',
                 'LANGUAGE', 'MONEY', 'PERCENT'],
    "required_packages": ["nltk", "JarbasModelZoo"]
}

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)


# corpus handling
def read_ontonotes5(filename):
    if filename.endswith(".ner"):
        with open(filename, 'r') as file_handle:
            file_content = file_handle.read()
            annotated_sentences = file_content.split('\n\n')
            if not annotated_sentences:
                return

            for annotated_sentence in annotated_sentences:
                annotated_tokens = [seq for seq in
                                    annotated_sentence.split('\n') if
                                    seq]
                if not annotated_tokens:
                    continue
                standard_form_tokens = []
                for idx, annotated_token in enumerate(
                        annotated_tokens):
                    annotations = annotated_token.split('\t')
                    word, tag, ner = annotations[0], annotations[1], \
                                     annotations[-1]
                    standard_form_tokens.append((word, tag, ner))

                # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                # Because the classfier expects a tuple as input, first item input, second the class
                yield [((w, t), iob) for w, t, iob in
                       standard_form_tokens]


corpus_root = "/home/user/my_code/OpenJarbas/nlp_models/OntoNotes-5.0-NER-BIO/onto.train.ner"
reader = read_ontonotes5(corpus_root)
training_samples = list(reader)

corpus_root = "/home/user/my_code/OpenJarbas/nlp_models/OntoNotes-5.0-NER-BIO/onto.test.ner"
reader = read_ontonotes5(corpus_root)
test_samples = list(reader)


def train():
    # training
    tagger = ClassifierBasedTagger(
        train=training_samples,
        feature_detector=extract_iob_features)

    # save pickle
    path = join(dirname(dirname(dirname(__file__))),
                "models", "ner", "nltk_clftagger_ontonotes5_NER.pkl")

    with open(path, "wb") as f:
        pickle.dump(tagger, f)


def test():
    path = join(dirname(dirname(dirname(__file__))),
                "models", "ner", "nltk_clftagger_ontonotes5_NER.pkl")
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
test()  # 0.9094512476681258
