# ModelZoo

trained models (with training scripts) for use across different projects

- [NLP Models](#nlp-models)
    * [NLTK](#nltk)
        + [Postag](#postag)
            - [Portuguese](#portuguese)
            - [Spanish](#spanish)
            - [Catalan](#catalan)

this package includes 2 utility methods to (down)load models, you can also use
the pickle library directly and skip installation of this package

|                  model_id                  | language |         dataset        |  task  | model type | accuracy |   required packages   |
|:------------------------------------------:|:--------:|:----------------------:|:------:|:----------:|:--------:|:---------------------:|
|    nltk_floresta_macmorpho_brill_tagger    |    pt    |  floresta + mac morpho | postag |   tagger   |          |          nltk         |
|          nltk_cess_es_brill_tagger         |    es    |         cess_es        | postag |   tagger   |          |          nltk         |
|          nltk_cess_ca_brill_tagger         |    ca    |         cess_ca        | postag |   tagger   |          |          nltk         |
|         nltk_treebank_brill_tagger         |    en    |        treebank        | postag |   tagger   |  0.9083  |          nltk         |
|         nltk_treebank_maxent_tagger        |    en    |        treebank        | postag |   tagger   |  0.9258  |          nltk         |
|           nltk_treebank_clftagger          |    en    |        treebank        | postag |   tagger   |          |                       |
|         nltk_treebank_ngram_tagger         |    en    |        treebank        | postag |   tagger   |          |                       |
|          nltk_treebank_tnt_tagger          |    en    |        treebank        | postag |   tagger   |          |                       |
|       nltk_conll2000_clf_chunk_tagger      |    en    |        conll2000       |  chunk |   tagger   |          |                       |
|         nltk_conll2000_clf_chunker         |    en    |        conll2000       |  chunk |   chunker  |          |                       |
|  nltk_conll2000_postag_ngram_chunk_tagger  |    en    |        conll2000       |  chunk |   tagger   |          |                       |
|     nltk_conll2000_postag_ngram_chunker    |    en    |        conll2000       |  chunk |   chunker  |          |                       |
| nltk_clftagger_paramopama+second_harem_NER |    pt    | paramopama + harem(v2) |   ner  |   tagger   |  0.8334  | nltk + JarbasModelZoo |
|        nltk_clftagger_paramopama_NER       |    pt    |       paramopama       |   ner  |   tagger   |  0.8396  | nltk + JarbasModelZoo |
|          nltk_clftagger_harem_NER          |    pt    |        harem(v1)       |   ner  |   tagger   |  0.9247  | nltk + JarbasModelZoo |
|        nltk_clftagger_miniharem_NER        |    pt    |       mini Harem       |   ner  |   tagger   |          |                       |
|         nltk_clftagger_leNERbr_NER         |    pt    |        leNER-Br        |   ner  |   tagger   |          |                       |
|           nltk_clftagger_gmb_NER           |    en    |           gmb          |   ner  |   tagger   |  0.9231  | nltk + JarbasModelZoo |
|        nltk_clftagger_conll2003_NER        |    en    |        conll2003       |   ner  |   tagger   |  0.9108  | nltk + JarbasModelZoo |
|          nltk_clftagger_WNUT17_NER         |    en    |         WNUT17         |   ner  |   tagger   |          |                       |

```bash
pip install JarbasModelZoo
```

# Security Concerns With the Python pickle Module

The serialization process is very convenient when you need to save your
object’s state to disk or to transmit it over a network.

However, there’s one more thing you need to know about the Python pickle
module: It’s not secure. the `__setstate__` method is great for doing more
initialization while unpickling, but it can also be used to execute arbitrary
code during the unpickling process!

So, what can you do to reduce this risk? Train the models yourself with the
provided scripts!

## Usage

### Postag

```python
from nltk import word_tokenize
from JarbasModelZoo import load_model

# will auto download if missing
# ~/.local/share/JarbasModelZoo/brill_tagger_floresta_mcmorpho_pt.pkl
tagger = load_model("brill_tagger_floresta_mcmorpho_pt")
tokens = word_tokenize("Olá, o meu nome é Joaquim")
postagged = tagger.tag(tokens)
# [('Olá', 'NOUN'), (',', '.'), ('o', 'DET'), ('meu', 'PRON'), ('nome', 'NOUN'), ('é', 'VERB'), ('Joaquim', 'NOUN')]

# ~/.local/share/JarbasModelZoo/brill_tagger_cess_es.pkl
tagger = load_model("brill_tagger_cess_es")
tokens = word_tokenize("Hola, mi nombre es Daniel")
postagged = tagger.tag(tokens)
# [('Hola', 'NOUN'), (',', 'fc'), ('mi', 'DET'), ('nombre', 'NOUN'), ('es', 'VERB'), ('Daniel', 'NOUN')]

# ~/.local/share/JarbasModelZoo/brill_tagger_cess_ca.pkl
tagger = load_model("brill_tagger_cess_ca")
tokens = word_tokenize("Quién es el presidente de Cataluña?")
postagged = tagger.tag(tokens)
# [('Quién', 'NOUN'), ('es', 'PRON'), ('el', 'DET'), ('presidente', 'NOUN'), ('de', 'ADP'), ('Cataluña', 'NOUN'), ('?', 'fit')]
```