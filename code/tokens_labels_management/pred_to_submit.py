import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm

class Prediction:
    def __init__(self, test_path, prediction_path):
        self.test = pd.read_csv(test_path)
        self.prediction = pd.read_csv(prediction_path)

    def to_submission(self):
        self.submission = self.test.copy()
        self.submission['label'] = 'O'
        sentences = self.test.sentence_id.unique()

        for sentence_id in tqdm(sentences):
            self.prediction_sentence = self.prediction[self.prediction.id_phrase//1000 == sentence_id]
            self.test_sentence = self.test[self.test.sentence_id == sentence_id]
            self.test_sentence['token'] = self.test_sentence['token'].str.replace('…', '...').replace('µ', 'μ')
            line_test = 0

            for index, line in self.prediction_sentence.iterrows():
                token = line['token']
                token_pred, token_id = self.test_sentence['token'].iloc[line_test], self.test_sentence['TokenId'].iloc[line_test]
                if ord(token[0]) == 9601:
                    if len(token) == 1:
                        self.submission['label'][token_id] = line['label']
                        line_test += 1
                        continue
                    cpt=0
                    while(token[1].lower() != token_pred[0].lower()):
                        cpt += 1
                        if cpt > 1 : 
                            print(index, token, token_pred, sentence_id)
                            break
                        line_test += 1
                        token_pred, token_id = self.test_sentence['token'].iloc[line_test], self.test_sentence['TokenId'].iloc[line_test]
                    self.submission['label'][token_id] = line['label']
                    line_test += 1
        return self.submission
