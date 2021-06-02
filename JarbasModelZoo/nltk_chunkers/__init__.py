from nltk import pos_tag, word_tokenize
from nltk.chunk import ChunkParserI
from nltk.chunk import RegexpParser
from nltk.chunk import conlltags2tree
from nltk.corpus import gazetteers

from JarbasModelZoo import load_model
from JarbasModelZoo.features import extract_iob_features


class NamedEntityChunker(ChunkParserI):
    def __init__(self, model_id):
        super().__init__()
        self.feature_detector = extract_iob_features
        self.tagger = load_model(model_id)

    def parse(self, tagged_sent):
        chunks = self.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)

    def tag(self, tagged_sent):
        if isinstance(tagged_sent, str):
            tagged_sent = pos_tag(word_tokenize(tagged_sent))

        return self.tagger.tag(tagged_sent)


class GazetteerLocationChunker(ChunkParserI):
    def __init__(self):
        self.locations = set(gazetteers.words())
        longest_word = max(self.locations, key=len)
        self.lookahead = len(longest_word.split(" "))

    def iob_locations(self, tagged_sent):
        if isinstance(tagged_sent, str):
            tagged_sent = pos_tag(word_tokenize(tagged_sent))

        found_locs = []
        for idx, (word, tag) in enumerate(tagged_sent):
            nexttags = tagged_sent[idx + 1:idx + self.lookahead]
            for i in range(len(nexttags)):
                loc_tags = nexttags[:i]
                loc_str = " ".join([word] + [n[0] for n in loc_tags])

                if loc_str in self.locations:
                    found_locs.append(idx)
                    yield word, tag, 'B-LOCATION'
                    for i, (w, t) in enumerate(loc_tags):
                        found_locs.append(i + idx + 1)
                        yield w, t, 'I-LOCATION'

                    break
            else:
                if word in self.locations:
                    found_locs.append(idx)
                    yield word, tag, 'B-LOCATION'

            if idx not in found_locs:
                yield word, tag, 'O'

    def tag(self, tagged_sent):
        return list(self.iob_locations(tagged_sent))

    def parse(self, tagged_sent):
        iobs = self.iob_locations(tagged_sent)
        return conlltags2tree(iobs)


class ProperNounChunker(ChunkParserI):
    def __init__(self):
        super().__init__()
        self.chunker = RegexpParser(r'''NAME:{<NNP>+}''')

    def parse(self, tagged_sent):
        if isinstance(tagged_sent, str):
            tagged_sent = pos_tag(word_tokenize(tagged_sent))
        return self.chunker.parse(tagged_sent)


class PartialRegexChunker(ChunkParserI):
    def __init__(self):
        super().__init__()
        self.chunker = RegexpParser(r'''
        NP:
            {<DT>?<NN.*>+}	# chunk optional determiner with nouns
            <JJ>{}<NN.*>	# merge adjective with noun chunk
        PP:
            {<IN>}			# chunk preposition
        VP: 
            {<MD>?<VB.*>}	# chunk optional modal with verb''')

    def parse(self, tagged_sent):
        if isinstance(tagged_sent, str):
            tagged_sent = pos_tag(word_tokenize(tagged_sent))
        return self.chunker.parse(tagged_sent)


class PostagChunkParser(ChunkParserI):
    def __init__(self, model_id, pos_tagger=None):
        self.chunker = load_model(model_id)
        if pos_tagger:
            self.tagger = load_model(pos_tagger)
        else:
            self.tagger = None

    def parse(self, sentence):
        if isinstance(sentence, str):
            if self.tagger:
                sentence = self.tagger.tag(word_tokenize(sentence))
            else:
                sentence = pos_tag(word_tokenize(sentence))
        pos_tags = [pos for word, pos in sentence]

        # Get the Chunk tags
        tagged_pos_tags = self.chunker.tag(pos_tags)

        # Assemble the (word, pos, chunk) triplets
        iob_triplets = [(word, pos_tag, chunk_tag)
                        for ((word, pos_tag), (pos_tag, chunk_tag)) in
                        zip(sentence, tagged_pos_tags)]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)


class ClassifierChunkParser(ChunkParserI):
    def __init__(self, model_id, pos_tagger=None):
        self.chunker = load_model(model_id)
        if pos_tagger:
            self.tagger = load_model(pos_tagger)
        else:
            self.tagger = None

    def parse(self, tagged_sent):
        if isinstance(tagged_sent, str):
            if self.tagger:
                tagged_sent = self.tagger.tag(word_tokenize(tagged_sent))
            else:
                tagged_sent = pos_tag(word_tokenize(tagged_sent))
        chunks = self.chunker.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)
