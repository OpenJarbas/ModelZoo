from JarbasModelZoo import load_model
from JarbasModelZoo.nltk_chunkers import NamedEntityChunker

ner = NamedEntityChunker(
    "/home/user/my_code/OpenJarbas/nlp_models/models/ner/nltk_clftagger_gmb_NER.pkl")
ner.parse("I'm going to Germany this Monday.")
# (S
#   I/PRP
#   'm/VBP
#   going/VBG
#   to/TO
#   (geo Germany/NNP)
#   this/DT
#   (tim Monday/NNP)
#   ./.)
ner.tag("I'm going to Germany this Monday.")
# [(('I', 'PRP'), 'O'), (("'m", 'VBP'), 'O'), (('going', 'VBG'), 'O'),
# (('to', 'TO'), 'O'), (('Germany', 'NNP'), 'B-geo'), (('this', 'DT'), 'O'),
# (('Monday', 'NNP'), 'B-tim'), (('.', '.'), 'O')]


chunker = load_model("nltk_gazetter_location_chunker")
chunker.tag([('San', 'NNP'), ('Francisco', 'NNP'),
             ('CA', 'NNP'), ('is', 'BE'), ('cold', 'JJ'),
             ('compared', 'VBD'), ('to', 'TO'), ('San', 'NNP'),
             ('Jose', 'NNP'), ('CA', 'NNP')])

# [('San', 'NNP', 'B-LOCATION'), ('Francisco', 'NNP', 'I-LOCATION'),
# ('CA', 'NNP', 'B-LOCATION'), ('is', 'BE', 'O'), ('cold', 'JJ', 'O'),
# ('compared', 'VBD', 'O'), ('to', 'TO', 'O'), ('San', 'NNP', 'B-LOCATION'),
# ('Jose', 'NNP', 'I-LOCATION'), ('CA', 'NNP', 'B-LOCATION')]
chunker.parse([('San', 'NNP'), ('Francisco', 'NNP'),
               ('CA', 'NNP'), ('is', 'BE'), ('cold', 'JJ'),
               ('compared', 'VBD'), ('to', 'TO'), ('San', 'NNP'),
               ('Jose', 'NNP'), ('CA', 'NNP')])
# (S
#   (LOCATION San/NNP Francisco/NNP)
#   (LOCATION CA/NNP)
#   is/BE
#   cold/JJ
#   compared/VBD
#   to/TO
#   (LOCATION San/NNP Jose/NNP)
#   (LOCATION CA/NNP))
