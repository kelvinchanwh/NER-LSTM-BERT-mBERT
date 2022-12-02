import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences

def read_wnut(url, test_data = False, wnut_2016 = False):
    if test_data:
      # Test data does not have labels
      df = pd.read_table(url, header=None, names=['token', 'upos'])
      df = df["token"]

    else:
      if wnut_2016:
        df = pd.read_table(url, header=None, names=['token', 'bio_only'], skip_blank_lines=False)
      else:
        df = pd.read_table(url, header=None, names=['token', 'label', 'bio_only', 'upos'])
      df = df[["token", "bio_only"]]

    # Split df based on tweet for easier feature extraction
    df_list = np.split(df, df[df.isnull().all(1)].index)
    # Remove NaN rows denoting seperation of tweets
    df_list = [sentence.dropna() for sentence in df_list]
    # Reset index of df for indexing
    df_list = [sentence.reset_index(drop=True) for sentence in df_list]

    return df_list

def pad_seq(tokenized_texts, tokenized_labels, tokenizer, tag2idx, MAX_LEN):
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=MAX_LEN, dtype="long", value=0.0,
                            truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in tokenized_labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")
    
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

    tensor_inputs = torch.tensor(input_ids)
    tensor_tags = torch.tensor(tags)
    tensor_masks = torch.tensor(attention_masks)

    return tensor_inputs, tensor_tags, tensor_masks

def tokenize_and_preserve_labels(sentence_list, tokenizer):
    sentence = [[word for word in sentence["token"]] for sentence in sentence_list]
    text_labels = [[s for s in sent["bio_only"]] for sent in sentence_list]

    tokenized_texts = list()
    tokenized_labels = list()
    
    for sent, labs in zip(sentence, text_labels):

        tokenized_sentence = []
        labels = []

        for word, label in zip(sent, labs):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(text = str(word))
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)
            
        tokenized_texts.append(tokenized_sentence)
        tokenized_labels.append(labels)

    return tokenized_texts, tokenized_labels


def preprocess(train_url, dev_url, test_url, model_name, MAX_LEN = 75, BATCH_SIZE = 32):
    train_list = read_wnut(train_url)
    dev_list = read_wnut(dev_url)
    test_list = read_wnut(test_url)

    unique_tags = list(set(tag for doc in train_list for tag in doc["labels"]))
    unique_tags.append("PAD")
    tag2idx = {t: i for i, t in enumerate(unique_tags)}

    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_tokenized_texts, train_tokenized_labels = tokenize_and_preserve_labels(train_list, tokenizer)
    dev_tokenized_texts, dev_tokenized_labels = tokenize_and_preserve_labels(dev_list, tokenizer)
    test_tokenized_texts, test_tokenized_labels = tokenize_and_preserve_labels(test_list, tokenizer)

    tr_inputs, tr_tags, tr_masks = pad_seq(train_tokenized_texts, train_tokenized_labels, tokenizer, tag2idx, MAX_LEN)
    dev_inputs, dev_tags, dev_masks = pad_seq(dev_tokenized_texts, dev_tokenized_labels, tokenizer, tag2idx, MAX_LEN)
    test_inputs, test_tags, test_masks = pad_seq(test_tokenized_texts, test_tokenized_labels, tokenizer, tag2idx, MAX_LEN)

    """The last step is to define the dataloaders. We shuffle the data at training time with the RandomSampler and at test time we just pass them sequentially with the SequentialSampler."""
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    dev_data = TensorDataset(dev_inputs, dev_masks, dev_tags)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)

    test_data = TensorDataset(test_inputs, test_masks, test_tags)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    train = [train_data, train_sampler, train_dataloader]
    dev = [dev_data, dev_sampler, dev_dataloader]
    test = [test_data, test_sampler, test_dataloader]

    return train, dev, test, unique_tags