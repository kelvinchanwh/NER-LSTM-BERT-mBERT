import LSTM.preprocessing
import LSTM.model
import BERT.preprocessing
import BERT.model
import eval
import tensorflow as tf
import numpy as np

wnuttrain = 'https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17train_clean_tagged.txt'
wnutdev = 'https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17dev_clean_tagged.txt'
wnuttest = "https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17test_annotated_clean_tagged.txt"


LSTM_dev_eval = eval.Evaluate()
LSTM_test_eval = eval.Evaluate()

for i in range(5):
    LSTM_train, LSTM_dev, LSTM_test, LSTM_token_vocab = LSTM.preprocessing.preprocess(wnuttrain, wnutdev, wnuttest)
    LSTM_dev_long, LSTM_test_long = LSTM.model.runLSTM(LSTM_train, LSTM_dev, LSTM_test, LSTM_token_vocab, EPOCHS = 100, BATCH_SIZE = 32)
    LSTM_dev_eval.evaluate(LSTM_dev_long, verbose = 0)
    LSTM_test_eval.evaluate(LSTM_test_long, verbose = 0)

print ("LSTM Development Set")
LSTM_dev_eval.print_results()
print ("LSTM Test Set")
LSTM_test_eval.print_results()


BERT_dev_eval = eval.Evaluate()
BERT_test_eval = eval.Evaluate()
for i in range(5):
    model_name = "bert-base-cased"
    BERT_train, BERT_dev, BERT_test, unique_tags = BERT.preprocessing.preprocess(wnuttrain, wnutdev, wnuttest, model_name, MAX_LEN = 75, BATCH_SIZE = 32)
    BERT_dev_long, BERT_test_long = BERT.model.runBERT(BERT_train, BERT_dev, BERT_test, model_name, unique_tags, epochs = 10, BATCH_SIZE = 32, output = 0)
    BERT_dev_eval.evaluate(BERT_dev_long, verbose = 0)
    BERT_test_eval.evaluate(BERT_test_long, verbose = 0)

print ("BERT Development Set")
BERT_dev_eval.print_results()
print ("BERT Test Set")
BERT_test_eval.print_results()

mBERT_dev_eval = eval.Evaluate()
mBERT_test_eval = eval.Evaluate()
for i in range(5):
    model_name = "bert-base-multilingual-cased"
    mBERT_train, mBERT_dev, mBERT_test, unique_tags = BERT.preprocessing.preprocess(wnuttrain, wnutdev, wnuttest, model_name, MAX_LEN = 75, BATCH_SIZE = 32)
    mBERT_dev_flat, mBERT_test_flat = BERT.model.runBERT(mBERT_train, mBERT_dev, mBERT_test, model_name, unique_tags, epochs = 10, BATCH_SIZE = 32, output = 0)
    mBERT_dev_eval.evaluate(mBERT_dev_flat, verbose = 0)
    mBERT_test_eval.evaluate(mBERT_test_flat, verbose = 0)

print ("mBERT Development Set")
mBERT_dev_eval.print_results()
print ("mBERT Test Set")
mBERT_test_eval.print_results()