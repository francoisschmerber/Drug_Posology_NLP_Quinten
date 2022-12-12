import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from transformers import CamembertTokenizerFast

import scripts.utils as utils

class JsonlDoccano:

    def __init__(self, filepath, camembert=True, shuffle_file_path='scripts/data/labels_augmentation.parquet', labels_augmentation=2, back_translation=False):
        if camembert:
            self.TOKENIZER = CamembertTokenizerFast.from_pretrained(
                'camembert-base',
                do_lower_case=True, encoding='utf-8').tokenize
        else:
            self.TOKENIZER = str.split
        
        self.camembert = camembert
            
        self.filepath = filepath
        self.shuffle_file_path = shuffle_file_path
        self.json_file = utils.read_jsonl(filepath=self.filepath, encoding='utf-8').to_pandas(self.TOKENIZER)
        self.df = self.to_pandas()
        self.labels = self.df[self.df['label'] != 'O']
        self.sentences = self.labels.id_phrase.unique()

        self.labels_augmentation = labels_augmentation
        self.back_translation = back_translation

    def to_pandas(self):
        list_df = []
        for entry in self.json_file:
            list_df.append(list(entry))
        df = pd.DataFrame(list_df, columns=['token', 'label', 'id_phrase'])
        return df

    def run_export(self, resultpath, augmented=False):
        if augmented:
            self.df_augmented.to_parquet(resultpath)
        else:
            self.df.to_parquet(resultpath)
        return
    
    def shuffle_labels(self, df_shuffle_tokens):
        
        json_file = utils.read_jsonl(filepath=self.filepath, encoding='utf-8').to_pandas(self.TOKENIZER)
        list_df = []
        remplace = False
        for entry in json_file:
            if entry[2] in self.sentences:
                if remplace and entry[1].startswith('I'):
                    continue
                remplace = False

                if entry[1].startswith('B'):
                    label = entry[1][2:]
                    if label in ['Drug', 'Form']:
                        line = df_shuffle_tokens[df_shuffle_tokens.label == label].sample(1)
                        for element in line['tokens'].to_numpy()[0]:
                            list_df.append(np.append(element, entry[2]))
                        remplace = True
                        continue

            list_df.append(list(entry))

        shuffled_df = pd.DataFrame(list_df, columns=['token', 'label', 'id_phrase'])
        return shuffled_df


    @property
    def df_augmented(self):
        shuffled_dfs = [self.df]
        df_shuffle_tokens = pd.read_parquet(self.shuffle_file_path)
        for i in range(self.labels_augmentation):
            shuffled_dfs.append(self.shuffle_labels(df_shuffle_tokens))
        if self.back_translation:
            shuffled_dfs.append(pd.read_parquet('scripts/data/back_translated.parquet'))
        df_augmented = pd.concat(shuffled_dfs)
        return df_augmented
                