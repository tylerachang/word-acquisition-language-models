# Modified from mikabr/aoa-prediction (Braginsky et al., 2016).
"""
Compiles data from the CHILDES dataset.
"""

import os
import nltk
from nltk.corpus.reader import CHILDESCorpusReader
from collections import defaultdict
import codecs
# from unicode_csv import *
# import feather

"""
The CHILDES corpus is available at ``https://childes.talkbank.org/``. The XML
version of CHILDES is located at ``https://childes.talkbank.org/data-xml/``.
Copy the needed parts of the CHILDES XML corpus into the NLTK data directory
(``nltk_data/corpora/CHILDES/``).
"""
# Takes a language, returns a CHIDLESCorpusReader for that language
def get_corpus_reader(language):
    return CHILDESCorpusReader(corpus_root, r'%s.*/.*\.xml' % language)

# Given a language, reads all CHILDES corpora in that language, computes
# word-level statistics (unigram counts, average sentence lengths, counts in
# sentence-final position, counts as sole constituent of an utterance)
def get_lang_data(language):
    corpus_reader = get_corpus_reader(language)
    # Later on, need the "/" characters to be removed.
    language = language.split("/")[-1]

    word_counts = defaultdict(float)
    sent_lengths = defaultdict(list)
    final_counts = defaultdict(float)
    solo_counts = defaultdict(float)

    # Each line will be a sentence, with words tab-separated.
    sents_outfile = codecs.open("childes_data/childes_{}.txt".format(language.lower()), 'w', encoding='utf-8')
    for corpus_file in corpus_reader.fileids():#[0:2]:
        # print "Getting data for %s" % corpus_file
        print("Reading {}".format(corpus_file))
        corpus_participants = corpus_reader.participants(corpus_file)[0]
        not_child = [value['id'] for key, value in corpus_participants.items() if key != 'CHI']

        # Can use lemmatized (stem) versions for consistency with previous papers.
        # Just change stem = True or False.
        # Note that this relies on stem tags in the XML, which is not always present.
        corpus_sents = corpus_reader.sents(corpus_file, speaker=not_child, replace=True, stem=False)
        for sent in corpus_sents:
            sentence_words = []
            for w in range(len(sent)):
                full_word = sent[w]
                word_separated = full_word.split('~')
                for word in word_separated:
                    word = word.split('-')[0].lower()
                    sentence_words.append(word)
                    word_counts[word] += 1
                    sent_lengths[word].append(len(sent))
                    if len(sent) == 1:
                        solo_counts[word] += 1
                    if w == len(sent) - 1:
                        final_counts[word] += 1
            formatted_sent = ' '.join(sentence_words).strip()
            if formatted_sent != '':
                sents_outfile.write("{}\n".format(formatted_sent))
    sents_outfile.close()

    mean_sent_lengths = {word: float(sum(word_sent_lengths)) / len(word_sent_lengths)
                         for word, word_sent_lengths in sent_lengths.items()}

    all_words = set(list(word_counts.keys()) + list(mean_sent_lengths.keys()) + list(final_counts.keys()) + list(solo_counts.keys()))
    word_data = {word: {"word_count": word_counts[word],
                        "mean_sent_length": mean_sent_lengths[word],
                        "final_count": final_counts[word],
                        "solo_count": solo_counts[word]} for word in all_words}

    outfile = codecs.open("childes_data/childes_{}.tsv".format(language.lower()), 'w', encoding='utf-8')
    outfile.write("word\tword_count\tmean_sent_length\tfinal_count\tsolo_count\n")
    for word, data in word_data.items():
        outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(word, data["word_count"], data["mean_sent_length"],
                                                        data["final_count"], data["solo_count"]))
    outfile.close()

corpus_root = nltk.data.find('corpora/childes/data-xml/')
print("Using root: {}".format(corpus_root))
languages = ["Eng-NA", "French", "Spanish", "Japanese", "German",
             "Chinese/Mandarin", "Spanish-ME", # Spanish-ME relies on manually separating the Mexican Spanish studies into a separate directory.
             "Romance/Italian", "Slavic/Croatian", "Slavic/Czech",
             "Slavic/Russian", "Chinese/Cantonese", "EastAsian/Korean",
             "Scandinavian/Danish", "Scandinavian/Norwegian", "Scandinavian/Swedish",
             "Other/Greek", "Other/Hebrew", "Other/Turkish"
             ]
for language in languages:
    get_lang_data(language)
