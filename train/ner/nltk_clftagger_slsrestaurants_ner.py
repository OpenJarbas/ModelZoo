import json
import pickle
from os.path import join, dirname

from nltk import pos_tag
from nltk.tag import ClassifierBasedTagger

from JarbasModelZoo.features import extract_iob_features
from JarbasModelZoo.nltk_chunkers import NamedEntityChunker, \
    conlltags2tree

MODEL_META = {
    "corpus": "MIT Restaurant Corpus",
    "lang": "en",
    "corpus_homepage": "https://groups.csail.mit.edu/sls/downloads/",
    "model_id": "nltk_clftagger_slsrestaurants_NER",
    "tagset": "conll_iob",
    "algo": "ClassifierBasedTagger | NamedEntityChunker",
    "entitíes": ['Restaurant_Name', 'Price', 'Cuisine', 'Hours', 'Amenity',
                 'Dish', 'Location', 'Rating'],
    "required_packages": ["nltk", "JarbasModelZoo"]
}

META = join(dirname(dirname(dirname(__file__))), "JarbasModelZoo", "res")
meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f)


# corpus handling
def read_sls(filename):
    if filename.endswith(".bio"):
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
                toks = [w.split('\t')[1] for w in annotated_tokens]
                pos = pos_tag(toks)
                for idx, annotated_token in enumerate(
                        annotated_tokens):
                    annotations = annotated_token.split('\t')

                    ner, word = annotations[0], annotations[1]
                    tag = pos[idx]
                    standard_form_tokens.append((word, tag, ner))

                # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                # Because the classfier expects a tuple as input, first item input, second the class
                yield [((w, t), iob) for w, t, iob in
                       standard_form_tokens]


corpus_file = "/home/user/my_code/OpenJarbas/nlp_models/SLS/mit_restaurants/restauranttrain.bio"
reader = read_sls(corpus_file)
training_samples = list(reader)

corpus_file = "/home/user/my_code/OpenJarbas/nlp_models/SLS/mit_restaurants/restauranttest.bio"
reader = read_sls(corpus_file)
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


train()
test()  # 0.8200757575757576
