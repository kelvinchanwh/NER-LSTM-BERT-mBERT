import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

def token_index(tok, token_vocab, oov):
    """
    Convert word tokens to integers
    """
    ind = tok
    if not pd.isnull(tok):  # new since last time: deal with the empty lines which we didn't drop yet
        if tok in token_vocab:  # if token in vocabulary
            ind = token_vocab.index(tok)
        else:  # else it's OOV
            ind = oov
    return ind

def bio_index(bio):
    """
    training labels: convert BIO to integers
    """
    ind = bio
    if not pd.isnull(bio):  # deal with empty lines
        if bio=='B':
            ind = 0
        elif bio=='I':
            ind = 1
        elif bio=='O':
            ind = 2
    return ind

def extract_features(txt, token_vocab, oov):
    """
    pass a data frame through our feature extractor
    """
    txt_copy = txt.copy()
    tokinds = [token_index(u, token_vocab, oov) for u in txt_copy['token']]
    txt_copy['token_indices'] = tokinds
    bioints = [bio_index(b) for b in txt_copy['bio_only']]
    txt_copy['bio_only'] = bioints
    return txt_copy

def tokens2sequences(txt_in):
    '''
    Takes panda dataframe as input, copies, and adds a sequence index based on full-stops.
    Outputs a dataframe with sequences of tokens, named entity labels, and token indices as lists.
    '''
    txt = txt_in.copy()
    txt['sequence_num'] = 0
    seqcount = 0
    for i in txt.index:  # in each row...
        txt.loc[i,'sequence_num'] = seqcount  # set the sequence number
        if pd.isnull(txt.loc[i,'token']):  # increment sequence counter at empty lines
            seqcount += 1
    # now drop the empty lines, group by sequence number and output df of sequence lists
    txt = txt.dropna()
    txt_seqs = txt.groupby(['sequence_num'],as_index=False)[['token', 'bio_only', 'token_indices']].agg(lambda x: list(x))
    return txt_seqs

def find_longest_sequence(txt,longest_seq):
    '''find the longest sequence in the dataframe'''
    for i in txt.index:
        seqlen = len(txt['token'][i])
        if seqlen > longest_seq:  # update high water mark if new longest sequence encountered
            longest_seq = seqlen
    return longest_seq

def parse_data(df, token_vocab, oov):
    df_copy = extract_features(df, token_vocab=token_vocab, oov = oov)
    df_seqs = tokens2sequences(df_copy)
    return df_seqs

def pad_sequence_and_labels(seqs, padtok, padlab, seq_length):
    # use pad_sequences, padding or truncating at the end of the sequence (default is 'pre')
    seqs_padded = tf.keras.preprocessing.sequence.pad_sequences(seqs['token_indices'].tolist(), maxlen=seq_length,
                                    dtype='int32', padding='post', truncating='post', value=padtok)

    # get lists of named entity labels, padded with a null label (=3)
    labs_padded = tf.keras.preprocessing.sequence.pad_sequences(seqs['bio_only'].tolist(), maxlen=seq_length,
                                    dtype='int32', padding='post', truncating='post', value=padlab)
    # convert those labels to one-hot encoding
    n_labs = padlab + 1  # we have 3 labels: B, I, O (0, 1, 2) + the pad label 3
    labs_onehot = [tf.keras.utils.to_categorical(i, num_classes=n_labs) for i in labs_padded]

    return seqs_padded, labs_padded, labs_onehot

def preprocess(train_url, dev_url, test_url):
    columns = ['token', 'label', 'bio_only', 'upos']
    train_df = pd.read_table(train_url, header=None, names=columns)
    dev_df = pd.read_table(dev_url, header=None, names=columns)
    test_df = pd.read_table(test_url, header=None, names=columns)

    token_vocab = train_df.token.unique().tolist()
    oov = len(token_vocab)  # OOV (out of vocabulary) token as vocab length (because that's max.index + 1)

    train_seqs = parse_data(train_df, token_vocab, oov)
    dev_seqs = parse_data(dev_df, token_vocab, oov)
    test_seqs = parse_data(test_df, token_vocab, oov)

    # set maximum sequence length
    longest = max(find_longest_sequence(train_seqs,0), find_longest_sequence(dev_seqs,0), find_longest_sequence(test_seqs,0))
    seq_length = longest

    # a new dummy token index, one more than OOV
    padtok = oov+1
    padlab = 3
    n_labs = padlab + 1  # we have 3 labels: B, I, O (0, 1, 2) + the pad label 3

    train_seqs_padded, train_labs_padded, train_labs_onehot = pad_sequence_and_labels(train_seqs, padtok, padlab, seq_length)
    dev_seqs_padded, dev_labs_padded, dev_labs_onehot = pad_sequence_and_labels(dev_seqs, padtok, padlab, seq_length)
    test_seqs_padded, test_labs_padded, test_labs_onehot = pad_sequence_and_labels(test_seqs, padtok, padlab, seq_length)

    train = [train_seqs, train_seqs_padded, train_labs_padded, train_labs_onehot]
    dev = [dev_seqs, dev_seqs_padded, dev_labs_padded, dev_labs_onehot]
    test = [test_seqs, test_seqs_padded, test_labs_padded, test_labs_onehot]

    return train, dev, test, token_vocab

