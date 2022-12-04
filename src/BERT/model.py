import pandas as pd
import numpy as np
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup, BertTokenizer
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm, trange
import torch


### Setup the Bert model for finetuning

def model_setup(model_name, tag2idx, total_steps):
    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = True
    )
    model.cuda()

    """
    Before we can start the fine-tuning process, we have to setup the optimizer and add the parameters it should update. A common choice is the AdamW optimizer. We also add some weight_decay as regularization to the main weight matrices. If you have limited resources, you can also try to just train the linear classifier on top of BERT and keep all other weights fixed. This will still give you a good performance."""

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=1e-5,
        eps=1e-8
    )

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    return model, optimizer, scheduler

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def trainModel(train, dev, model, tokenizer, optimizer, scheduler, epochs, unique_tags):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = train[0]
    train_sampler = train[1]
    train_dataloader = train[2]

    dev_data = dev[0]
    dev_sampler = dev[1]
    dev_dataloader = dev[2]

    early_stopper = EarlyStopper(patience = 5, min_delta=0)
    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []
    max_grad_norm = 1.0
    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)

            # logits = outputs[1]
            # get the loss
            loss = outputs[0]

            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print ()
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        val_pred_tags, val_valid_tags, val_token_ids = [], [], []

        for batch in dev_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            eval_accuracy += flat_accuracy(logits, label_ids)
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)
            val_token_ids.extend(b_input_ids)

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        for p, l in zip(predictions, true_labels):
            val_pred_tags_sent = [unique_tags[p_i] for p_i, l_i in zip(p, l) if unique_tags[l_i] != "PAD"]
            val_valid_tags_sent = [unique_tags[l_i] for l_i in l if unique_tags[l_i] != "PAD"]
            val_pred_tags.append(val_pred_tags_sent)
            val_valid_tags.append(val_valid_tags_sent)

        val_tokens = [tokenizer.convert_ids_to_tokens(tok) for tok in val_token_ids]

        val_pred_tags_flat = [tag[i] for tag in val_pred_tags for i in range(len(tag))]
        val_valid_tags_flat = [tag[i] for tag in val_valid_tags for i in range(len(tag))]
        val_tokens_flat = [tok for sent in val_tokens for tok in sent if tok != "[PAD]"]

        print("Validation F1-Score: {}".format(f1_score(val_valid_tags_flat, val_pred_tags_flat, labels = ["B", "I"], average="weighted")))
        print()

        if early_stopper.early_stop(eval_loss):
            return val_pred_tags, val_valid_tags, val_tokens
            break
        return val_pred_tags_flat, val_valid_tags_flat, val_tokens_flat
        

def predict(test, model, tokenizer, unique_tags):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ========================================
    #               Test Set
    # ========================================
    # 
    test_data = test[0]
    test_sampler = test[1]
    test_dataloader = test[2]

    # Put the model into evaluation mode
    model.eval()
    # Reset the test loss.
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0
    predictions , true_labels = [], []
    pred_tags, valid_tags, token_ids = [], [], []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask) #, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        # test_loss += outputs[0].mean().item()
        test_accuracy += flat_accuracy(logits, label_ids)
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)
        token_ids.extend(b_input_ids)

        nb_test_examples += b_input_ids.size(0)
        nb_test_steps += 1

    test_loss = test_loss / nb_test_steps
    print("Test loss: {}".format(test_loss))
    print("Test Accuracy: {}".format(test_accuracy/nb_test_steps))

    for p, l in zip(predictions, true_labels):
        pred_tags_sent = [unique_tags[p_i] for p_i, l_i in zip(p, l) if unique_tags[l_i] != "PAD"]
        valid_tags_sent = [unique_tags[l_i] for l_i in l if unique_tags[l_i] != "PAD"]

        pred_tags.append(pred_tags_sent)
        valid_tags.append(valid_tags_sent)

    pred_tags_flat = [tag[i] for tag in pred_tags for i in range(len(tag))]
    valid_tags_flat = [tag[i] for tag in valid_tags for i in range(len(tag))]

    tokens = [tokenizer.convert_ids_to_tokens(tok) for tok in token_ids]
    tokens_flat = [tok for sent in tokens for tok in sent if tok != "[PAD]"]

    print("Test F1-Score: {}".format(f1_score(valid_tags_flat, pred_tags_flat, labels = ["B", "I"], average="weighted")))
    print()

    test_valid_tags_flat = valid_tags_flat.copy()
    test_pred_tags_flat = pred_tags_flat.copy()
    test_tokens_flat = tokens_flat.copy()

    return test_valid_tags_flat, test_pred_tags_flat, test_tokens_flat

def runBERT(train, dev, test, model_name, unique_tags, epochs = 10, BATCH_SIZE = 32, output = 0):
    tag2idx = {t: i for i, t in enumerate(unique_tags)}
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train[2]) * epochs
    model, optimizer, scheduler = model_setup(model_name, tag2idx, total_steps)

    dev_pred_tags_flat, dev_valid_tags_flat, dev_tokens_flat = trainModel(train, dev, model, tokenizer, optimizer, scheduler, epochs, unique_tags)
    test_valid_tags_flat, test_pred_tags_flat, test_tokens_flat = predict(test, model, tokenizer, unique_tags)

    dev = pd.DataFrame()
    dev["tokens"] = dev_tokens_flat
    dev["bio_only"] = dev_valid_tags_flat
    dev["prediction"] = dev_pred_tags_flat
    
    test = pd.DataFrame()
    test["tokens"] = test_tokens_flat
    test["bio_only"] = test_valid_tags_flat
    test["prediction"] = test_pred_tags_flat

    return dev, test


