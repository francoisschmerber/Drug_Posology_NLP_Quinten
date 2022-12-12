
import torch
import numpy as np
import pandas as pd

SENTENCES_SIZE = 128
TOKENS_NUMBER = 256



def transform_label(x):
    if x != 'O':
        return x[2:]
    else:
        return x


def truncate(df, size=SENTENCES_SIZE):
    list_rajout = []
    for index, line in df.iterrows():
        tokens, tags = line.tokens, line.ner_tags
        while len(tokens) > size:
            list_rajout.append([line.id_phrase, tokens[:size], tags[:size]])
            tokens, tags = tokens[size:], tags[size:]
        list_rajout.append([line.id_phrase, tokens, tags])

    return pd.DataFrame(list_rajout, columns = ['id_phrase', 'tokens', 'ner_tags'])


def align_labels_with_tokens(labels, word_ids, tag2id):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else tag2id[labels[word_id]]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = tag2id[labels[word_id]]
            # If the label is B-XXX we change it to I-XXX
            #if label % 2 == 1:
                #label += 1
            new_labels.append(label)

    return new_labels


def encode_tags(tags, encodings, tag2id):
    encoded_labels = []
    for i, ner_tags in enumerate(tags):
        encoded_labels.append(align_labels_with_tokens(ner_tags, encodings[i].word_ids), tag2id)

    return encoded_labels

class QuintenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def evaluate_model(pathmodel, val_dataset, val_labels, id2tag):
    model = torch.load(pathmodel)

    input_ids, attention_masks = val_dataset.encodings['input_ids'], val_dataset.encodings['attention_mask']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    flat_pred = []
    with torch.no_grad():
        input_ids, attention_masks = torch.tensor(input_ids), torch.tensor(attention_masks)

        outputs =  model(input_ids.to(device), token_type_ids=None, attention_mask=attention_masks.to(device))

    logits = outputs['logits'].detach().cpu().numpy()
    pred = np.argmax(logits, axis=2).flatten()
    true_labels = np.array(val_labels).flatten()

    liste_results = []
    for i in range(len(pred)):
        if id2tag[pred[i]] != 'O' or (true_labels[i] not in [-100, 0]):
            liste_results.append(pred[i]==true_labels[i])
    print( "Sur l'ensemble des vrais labels différents de O et des labels prédits différents de O, "+"{:.1f}".format(np.mean(liste_results)*100)+"% ont le bon label prédit")


def predict_test(df, model, id2tag, tokenizer):

    encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding='max_length',
                            truncation='longest_first', max_length=TOKENS_NUMBER
                            )

    input_ids, attention_masks = encodings['input_ids'], encodings['attention_mask']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    flat_pred = []
    with torch.no_grad():
        input_ids, attention_masks = torch.tensor(input_ids), torch.tensor(attention_masks)

        outputs =  model(input_ids.to(device), token_type_ids=None, attention_mask=attention_masks.to(device))

    logits = outputs['logits'].detach().cpu().numpy()
    pred = np.argmax(logits, axis=2).flatten()
    id_sentences = df.id_phrase.to_numpy()
    sentences = np.array([[id_sentence]*256 for id_sentence in id_sentences]).flatten()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy().flatten())
    list_results = []
    for i in range(len(pred)):
        list_results.append([tokens[i], id2tag[pred[i]]])
    results = pd.DataFrame(list_results, columns=['token', 'label'])  
    results['id_phrase'] = [sentence*1000 for sentence in sentences]

    result_final = results[~results.token.str.startswith('<')]

    return result_final