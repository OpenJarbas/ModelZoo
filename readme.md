# ModelZoo

trained models (with training scripts) for use across different projects

```bash
pip install JarbasModelZoo
```

# Models

this package includes utility methods to (down)load models

training scripts can be found in the [train folder](./train)

### NER

|model_id|language|dataset|accuracy|
|:------:|:------:|:------:|:------:|
|nltk_clftagger_conll2003_NER|en|[CONLL2003](https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003)| 0.874% |
|nltk_clftagger_gmb_NER|en|[GMB 2.2.0](http://gmb.let.rug.nl/data.php)| 0% |
|nltk_clftagger_slsmovies_NER|en|[MIT Movie Corpus](https://groups.csail.mit.edu/sls/downloads/)| 0% |
|nltk_clftagger_slstrivia10k13_NER|en|[MIT Movie Corpus - Trivia](https://groups.csail.mit.edu/sls/downloads/)| 0.806% |
|nltk_clftagger_slsrestaurants_NER|en|[MIT Restaurant Corpus](https://groups.csail.mit.edu/sls/downloads/)| 0% |
|nltk_clftagger_onto5_NER|en|[OntoNotes-5.0-NER-BIO](https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO)| 0.910% |
|nltk_clftagger_paramopama_NER|pt|[Paramopama](https://github.com/davidsbatista/NER-datasets/blob/master/Portuguese/Paramopama)| 0% |
|nltk_clftagger_paramopama+harem_NER|pt|[Paramopama + HAREM (v2)](https://github.com/davidsbatista/NER-datasets/blob/master/Portuguese/Paramopama)| 0% |
|nltk_clftagger_WNUT17_NER|en|[WNUT17](https://github.com/leondz/emerging_entities_17)| 0% |
|nltk_clftagger_leNERbr_NER|pt-br|[leNER-Br](https://cic.unb.br/~teodecampos/LeNER-Br/)| 0% |

### POSTAG

|model_id|language|dataset|tagset|accuracy|
|:------:|:------:|:------:|:------:|:------:|
|nltk_floresta_macmorpho_brill_tagger|pt|floresta + macmorpho|universal| 0% |
|nltk_brown_brill_tagger|en|[brown](http://www.hit.uib.no/icame/brown/bcm.html)|brown| 0.941% |
|nltk_brown_maxent_tagger|en|[brown](http://www.hit.uib.no/icame/brown/bcm.html)|brown| 0% |
|nltk_brown_ngram_tagger|en|[brown](http://www.hit.uib.no/icame/brown/bcm.html)|brown| 0.930% |
|nltk_floresta_brill_tagger|pt|[floresta](http://www.linguateca.pt/Floresta)|[VISL (Portuguese)](https://visl.sdu.dk/visl/pt/symbolset-floresta.html)| 0.938% |
|nltk_floresta_ngram_tagger|pt|[floresta](http://www.linguateca.pt/Floresta)|[VISL (Portuguese)](https://visl.sdu.dk/visl/pt/symbolset-floresta.html)| 0.925% |
|nltk_cess_cat_udep_brill_tagger|ca|[cess_cat_udep](https://github.com/OpenJarbas/biblioteca/blob/master/corpora/create_cess_ca.py)|Universal Dependencies| 0.974% |
|nltk_cess_esp_udep_brill_tagger|es|[cess_esp_udep](https://github.com/OpenJarbas/biblioteca/blob/master/corpora/create_cess.py)|Universal Dependencies| 0.975% |
|nltk_macmorpho_unvtagset_brill_tagger|pt|[macmorpho](http://www.nilc.icmc.usp.br/lacioweb/)|Universal Dependencies| 0.966% |
|nltk_onto5_brill_tagger|en|[OntoNotes-5.0-NER-BIO](https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO)|Penn Treebank| 0% |
|nltk_treebank_clftagger|en|treebank|Penn Treebank| 0% |
|nltk_treebank_brill_tagger|en|treebank|Penn Treebank| 0% |
|nltk_treebank_ngram_tagger|en|treebank|Penn Treebank| 0% |
|nltk_treebank_maxent_tagger|en|treebank|Penn Treebank| 0% |
|nltk_treebank_tnt_tagger|en|treebank|Penn Treebank| 0% |
|nltk_nilc_brill_tagger|pt-br|[NILC_taggers](http://www.nilc.icmc.usp.br/nilc/tools/nilctaggers.html)|[NILC](http://www.nilc.icmc.usp.br/nilc/download/tagsetcompleto.doc)| 0.881% |
|nltk_nilc_ngram_tagger|pt-br|[NILC_taggers](http://www.nilc.icmc.usp.br/nilc/tools/nilctaggers.html)|[NILC](http://www.nilc.icmc.usp.br/nilc/download/tagsetcompleto.doc)| 0.869% |
|nltk_cess_cat_brill_tagger|ca|[cess_cat](https://web.archive.org/web/20121023154634/http://clic.ub.edu/cessece/)|[EAGLES](http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html)| 0.939% |
|nltk_cess_esp_brill_tagger|es|cess_esp|[EAGLES](http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html)| 0.926% |
|nltk_macmorpho_brill_tagger|pt|macmorpho|| 0% |

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