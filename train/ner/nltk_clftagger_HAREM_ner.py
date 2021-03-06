import pickle
import random
from os.path import join, dirname

from nltk.tag import ClassifierBasedTagger

from JarbasModelZoo import load_model
from JarbasModelZoo.features import extract_iob_features
from JarbasModelZoo.nltk_chunkers import NamedEntityChunker, \
    conlltags2tree

MODEL_META = {
    "corpus": "HAREM",
    "lang": "pt",
    "model_id": "nltk_clftagger_harem_NER",
    "tagset": "conll_iob",
    "algo": "ClassifierBasedTagger | NamedEntityChunker",
    "entitíes": ['ORGANIZACAO', 'VALOR', 'TEMPO', 'LOCAL', 'PESSOA'],
    "required_packages": ["nltk", "JarbasModelZoo"]
}


# corpus handling
def to_conll_iob(annotated_sentence):
    """
    `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token

        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))
    return proper_iob_tokens


def postag_corpus(corpus_root,
                  postagger="/home/user/my_code/OpenJarbas/nlp_models/models/postag/nltk_floresta_macmorpho_brill_tagger.pkl"):
    tagger = load_model(postagger)
    tags = []

    with open(corpus_root, 'r') as file_handle:
        file_content = file_handle.read().strip()
        annotated_sentences = file_content.split('\n\n')

        for annotated_sentence in annotated_sentences:
            annotated_tokens = [seq for seq in
                                annotated_sentence.split('\n') if
                                seq]
            words = [w.split('\t')[0] for w in annotated_tokens]
            tagged = tagger.tag(words)
            standard_form_tokens = []
            for idx, annotated_token in enumerate(annotated_tokens):
                annotations = annotated_token.split('\t')
                word, ner = annotations[0], annotations[1]
                tag = tagged[idx][1]
                if ner != 'O':
                    ner = ner.split('-')[0]
                    tags.append(ner)

                standard_form_tokens.append((word, tag, ner))

            conll_tokens = to_conll_iob(standard_form_tokens)

            # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
            # Because the classfier expects a tuple as input, first item input, second the class
            if conll_tokens:
                yield [((w, t), iob) for w, t, iob in conll_tokens]

    print(list(set(tags)))


# Download the corpus here:
# https://github.com/davidsbatista/NER-datasets/blob/master/Portuguese/Paramopama
corpus_root = "/home/user/my_code/OpenJarbas/nlp_models/NER-datasets/Portuguese/Paramopama/corpus_First_HAREM.txt"
reader = postag_corpus(corpus_root)
training_samples = list(reader)
random.shuffle(training_samples)
corpus_root = "/home/user/my_code/OpenJarbas/nlp_models/NER-datasets/Portuguese/Paramopama/corpus_miniHAREM.txt"
reader = postag_corpus(corpus_root)
test_samples = list(reader)
random.shuffle(test_samples)


def train():
    # training
    tagger = ClassifierBasedTagger(
        train=training_samples,
        feature_detector=extract_iob_features)

    # save pickle
    path = join(dirname(dirname(dirname(__file__))),
                "models", "ner", "nltk_clftagger_harem_NER.pkl")

    with open(path, "wb") as f:
        pickle.dump(tagger, f)


def test():
    path = join(dirname(dirname(dirname(__file__))),
                "models", "ner", "nltk_clftagger_harem_NER.pkl")
    chunker = NamedEntityChunker(path)
    # accuracy test
    score = chunker.evaluate(
        [conlltags2tree([(w, t, iob) for (w, t), iob in iobs])
         for iobs in test_samples])
    print(score.accuracy())


train()
test()  # 0.8637345012645734
