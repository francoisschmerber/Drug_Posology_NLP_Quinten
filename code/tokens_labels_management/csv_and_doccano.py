import json
import pandas as pd

class CsvDoccano:

    def __init__(self, filepath='data/test.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

    def to_doccano(self, direcpath='test.jsonl'):
        parag = self.df.sentence_id.unique()
        with open(direcpath, mode='w', encoding='utf-8') as test :
            for sentence_id in parag:
                dico = {}
                dico = {"id":str(sentence_id)}
                dico["text"] = " ".join(self.df[self.df.sentence_id==sentence_id]['token']) + "\n"
                dico['labels'] = []
                test.write(json.dumps(dico, ensure_ascii=False) + "\n")
        return

