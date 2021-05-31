from os.path import join, dirname

MODEL = join(dirname(dirname(dirname(__file__))), "models", "postag")

import nltk
from random import shuffle
from os.path import join, dirname, isfile
from string import punctuation
import pickle


def train_pt_tagger(path):
    nltk.download('mac_morpho')
    nltk.download('floresta')

    def convert_to_universal_tag(t, reverse=False):
        tagdict = {
            'n': "NOUN",
            'num': "NUM",
            'v-fin': "VERB",
            'v-inf': "VERB",
            'v-ger': "VERB",
            'v-pcp': "VERB",
            'pron-det': "PRON",
            'pron-indp': "PRON",
            'pron-pers': "PRON",
            'art': "DET",
            'adv': "ADV",
            'conj-s': "CONJ",
            'conj-c': "CONJ",
            'conj-p': "CONJ",
            'adj': "ADJ",
            'ec': "PRT",
            'pp': "ADP",
            'prp': "ADP",
            'prop': "NOUN",
            'pro-ks-rel': "PRON",
            'proadj': "PRON",
            'prep': "ADP",
            'nprop': "NOUN",
            'vaux': "VERB",
            'propess': "PRON",
            'v': "VERB",
            'vp': "VERB",
            'in': "X",
            'prp-': "ADP",
            'adv-ks': "ADV",
            'dad': "NUM",
            'prosub': "PRON",
            'tel': "NUM",
            'ap': "NUM",
            'est': "NOUN",
            'cur': "X",
            'pcp': "VERB",
            'pro-ks': "PRON",
            'hor': "NUM",
            'pden': "ADV",
            'dat': "NUM",
            'kc': "ADP",
            'ks': "ADP",
            'adv-ks-rel': "ADV",
            'npro': "NOUN",
        }
        if t in ["N|AP", "N|DAD", "N|DAT", "N|HOR", "N|TEL"]:
            t = "NUM"
        if reverse:
            if "|" in t: t = t.split("|")[0]
        else:
            if "+" in t: t = t.split("+")[1]
            if "|" in t: t = t.split("|")[1]
            if "#" in t: t = t.split("#")[0]
        t = t.lower()
        return tagdict.get(t, "." if all(tt in punctuation for tt in t) else t)

    floresta = [[(w, convert_to_universal_tag(t))
                 for (w, t) in sent]
                for sent in nltk.corpus.floresta.tagged_sents()]
    shuffle(floresta)

    mac_morpho = [[w[0] for w in sent] for sent in
                  nltk.corpus.mac_morpho.tagged_paras()]
    mac_morpho = [
        [(w, convert_to_universal_tag(t, reverse=True))
         for (w, t) in sent] for sent in mac_morpho]
    shuffle(mac_morpho)

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

    def_tagger = nltk.DefaultTagger('NOUN')
    affix_tagger = nltk.AffixTagger(
        mac_morpho + floresta, backoff=def_tagger
    )
    unitagger = nltk.UnigramTagger(
        mac_morpho + floresta, backoff=affix_tagger
    )
    rx_tagger = nltk.RegexpTagger(
        regex_patterns, backoff=unitagger
    )
    tagger = nltk.BigramTagger(
        floresta, backoff=rx_tagger
    )
    tagger = nltk.BrillTaggerTrainer(tagger, nltk.brill.fntbl37())
    tagger = tagger.train(floresta, max_rules=100)

    with open(path, "wb") as f:
        pickle.dump(tagger, f)

    return tagger



train_pt_tagger(join(MODEL, "brill_tagger_floresta_mcmorpho_pt.pkl"))
