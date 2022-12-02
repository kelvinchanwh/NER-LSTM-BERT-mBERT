import tensorflow as tf
from collections import Counter
import copy
import numpy as np
import pandas as pd

import preprocessing


def calc_init_bias(train_labs_padded):
    # figure out the label distribution in our fixed-length texts
    all_labs = [l for lab in train_labs_padded for l in lab]
    label_count = Counter(all_labs)
    total_labs = len(all_labs)

    # use this to define an initial model bias
    initial_bias=[(label_count[0]/total_labs), (label_count[1]/total_labs),
                  (label_count[2]/total_labs), (label_count[3]/total_labs)]
    print('Initial bias:')
    print(initial_bias)

def make_model(metrics, output_bias, vocab_size, embed_size, seq_length, n_labs):
    if output_bias is not None:
        output_bias = tf.tf.keras.initializers.Constant(output_bias)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=seq_length, mask_zero=True, trainable=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),  # 2 directions, 50 units each, concatenated (can change this)
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_labs, activation='softmax', bias_initializer=output_bias)),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)
    return model

def get_training_weights(train_labs_onehot):
    train_weights_onehot = copy.deepcopy(train_labs_onehot)

    # our first-pass class weights: normal for named entities (0 and 1), down-weighted for non named entities (2 and 3)
    class_wts = [1,1,.005,.005]

    # apply our weights to the label lists
    for i,labs in enumerate(train_weights_onehot):
    	for j,lablist in enumerate(labs):
            lablistaslist = lablist.tolist()
            whichismax = lablistaslist.index(max(lablistaslist))
            train_weights_onehot[i][j][whichismax] = class_wts[whichismax]
    return train_labs_onehot

def reverse_bio(ind):
    bio = 'O'  # for any pad=3 predictions
    if ind==0:
      bio = 'B'
    elif ind==1:
      bio = 'I'
    elif ind==2:
      bio = 'O'
    return bio

def get_prediction(model, df_seqs, df_seqs_padded):
    df_preds = np.argmax(model.predict(df_seqs_padded), axis=-1)
    df_flat_preds = [p for pred in df_preds for p in pred]

    # start a new column for the model predictions
    df_seqs['prediction'] = ''

    # for each text: get original sequence length and trim predictions accordingly
    # (_trim_ because we know that our seq length is longer than the longest seq in dev)
    for i in df_seqs.index:
      this_seq_length = len(df_seqs['token'][i])
      df_seqs['prediction'][i] = df_preds[i][:this_seq_length].astype(int)

    """Then we need to reshape the sequences back into long tabular format ready for our evaluation function."""

    # use sequence number as the index and apply pandas explode to all other columns
    df_long = df_seqs.set_index('sequence_num').apply(pd.Series.explode).reset_index()

    """Finally we need to convert the named entity labels from integers back to BIO characters:"""

    df_long['bio_only'] = [reverse_bio(b) for b in df_long['bio_only']]
    df_long['prediction']  = [reverse_bio(b) for b in df_long['prediction']]

    return df_long


def runLSTM(train, dev, test, token_vocab, EPOCHS = 100, BATCH_SIZE = 32, output = 0):
    train_seqs = train[0]
    train_seqs_padded = train[1]
    train_labs_padded = train[2]
    train_labs_onehot = train[3]

    dev_seqs = dev[0]
    dev_seqs_padded = dev[1]
    dev_labs_padded = dev[2]
    dev_labs_onehot = dev[3]

    test_seqs = test[0]
    test_seqs_padded = test[1]
    test_labs_padded = test[2]
    test_labs_onehot = test[3]


    # prepare sequences and labels as numpy arrays, check dimensions
    train_X = np.array(train_seqs_padded)
    train_y = y = np.array(get_training_weights(train_labs_onehot)) # Weighted labels

    # prepare the dev sequences and labels as numpy arrays
    dev_X = np.array(dev_seqs_padded)
    dev_y = np.array(dev_labs_onehot)

    # prepare the test sequences and labels as numpy arrays
    test_X = np.array(test_seqs_padded)
    test_y = np.array(test_labs_onehot)

    # our final vocab size is the padding token + 1 (OR length of vocab + OOV + PAD)
    seq_length = max(preprocessing.find_longest_sequence(train_seqs,0), preprocessing.find_longest_sequence(dev_seqs,0), preprocessing.find_longest_sequence(test_seqs,0))
    oov = len(token_vocab)  # OOV (out of vocabulary) token as vocab length (because that's max.index + 1)
    embed_size = 128  # try an embedding size of 128 (could tune this)
    padtok = oov+1
    vocab_size = padtok+1
    padlab = 3
    n_labs = padlab + 1

    # list of metrics to use: true & false positives, negatives, accuracy, precision, recall, area under the curve
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', verbose=1, patience=10, mode='max', restore_best_weights=True)

    initial_bias = calc_init_bias(train_labs_padded)
    model = make_model(METRICS, initial_bias, vocab_size, embed_size, seq_length, n_labs)
    model.fit(train_X, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [early_stopping], validation_data=(dev_X, dev_y))

    dev_long = get_prediction(model, dev_seqs, dev_seqs_padded)
    test_long = get_prediction(model, test_seqs, test_seqs_padded)

    if output:
        dev_long.to_csv("LSTM_Dev.csv")
        test_long.to_csv("LSTM_Test.csv")

    return dev_long, test_long
