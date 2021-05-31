# NLP Models

trained models (with training scripts) for use across different projects

- [NLP Models](#nlp-models)
  * [NLTK](#nltk)
    + [Postag](#postag)
      - [Portuguese](#portuguese)
      - [Spanish](#spanish)
      - [Catalan](#catalan)
    
## NLTK

### Postag

nltk does not come with pre-trained pos taggers for most languages

#### Portuguese

brill tagger trained on floresta and mac_morpho corpus

```python
import pickle
from nltk import word_tokenize

with open("brill_tagger_floresta_mcmorpho_pt.pkl", "rb") as f:
    tagger = pickle.load(f)
    
tokens = word_tokenize("Olá, o meu nome é Joaquim")
postagged = tagger.tag(tokens)
# [('Olá', 'NOUN'), (',', '.'), ('o', 'DET'), ('meu', 'PRON'), ('nome', 'NOUN'), ('é', 'VERB'), ('Joaquim', 'NOUN')]
```

#### Spanish

brill tagger trained on cess_esp corpus

```python
import pickle
from nltk import word_tokenize

with open("brill_tagger_cess_es.pkl", "rb") as f:
    tagger = pickle.load(f)
    
tokens = word_tokenize("Hola, mi nombre es Daniel")
postagged = tagger.tag(tokens)
# [('Hola', 'NOUN'), (',', 'fc'), ('mi', 'DET'), ('nombre', 'NOUN'), ('es', 'VERB'), ('Daniel', 'NOUN')]
```

#### Catalan

brill tagger trained on cess_esp corpus

```python
import pickle
from nltk import word_tokenize

with open("brill_tagger_cess_ca.pkl", "rb") as f:
    tagger = pickle.load(f)
    
tokens = word_tokenize("Quién es el presidente de Cataluña?")
postagged = tagger.tag(tokens)
# [('Quién', 'NOUN'), ('es', 'PRON'), ('el', 'DET'), ('presidente', 'NOUN'), ('de', 'ADP'), ('Cataluña', 'NOUN'), ('?', 'fit')]
```