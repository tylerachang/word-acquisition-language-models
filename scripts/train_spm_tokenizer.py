# Use vocab size 30,000 as in BERT.
# To train faster, randomly sampling 10,000,000 sentences (lines).
# The input data file should be the input file for the LineByLineTextDataset (raw text).
import sentencepiece as spm
spm.SentencePieceTrainer.train(input='spm_temp/cleaned_wiki103_bookcorpus_train.txt',
        model_prefix='spm_temp/wiki103_bookcorpus_spm', vocab_size=30000,
        input_sentence_size=10000000, shuffle_input_sentence=True)
