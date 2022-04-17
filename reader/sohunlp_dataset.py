import json
import logging
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, input_file, max_len, tokenizer):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self._load_data(input_file)
    
    def _load_data(self, input_file):
        self.texts = []
        self.labels = []
        self.entitys = []
        self.sample_ids = []
        bad_lines = 0
        with open(input_file) as f:
            for line in f:
                splits = line.strip().split('\t')
                if len(splits) != 4:
                    bad_lines += 1
                    continue
                sid, label, entity, text = splits
                self.sample_ids.append(sid)
                self.entitys.append(entity)
                self.texts.append(text)
                self.labels.append(int(label) + 2)
                # self.labels.append(int(label))
        self.n_classes = len(set(self.labels))
        logging.info("%s bad lines" % bad_lines)

    def __len__(self):
        return len(self.entitys)

    def __getitem__(self, index):
        entity = self.entitys[index]
        label = self.labels[index]
        text = self.texts[index]
        sent = entity + 'SEP' + text
        
        encoded_sent = self.tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=self.max_len,                  # Max length to truncate/pad
            padding='max_length',
            truncation=True,
            return_attention_mask=True,      # Return attention mask
        )
        item = {
            'input_ids': encoded_sent['input_ids'],
            'attention_mask': encoded_sent['attention_mask'],
            'label': label
        }
        return item
    
    def gen_submission(self, preds, output_path):
        res_dict = {}
        for idx, label in enumerate(preds):
            sample_id = self.sample_ids[idx]
            entity_name = self.entitys[idx]
            if sample_id not in res_dict:
                res_dict[sample_id] = {}
            res_dict[sample_id][entity_name] = label - 2
        wr = open(output_path, 'w')
        wr.write('id\tresult\n')
        for sample_id, res in sorted(res_dict.items(), key=lambda x: x[0]):
            #print(sample_id)
            #print(res)
            wr.write(str(sample_id) + '\t' + json.dumps(res, ensure_ascii=False) + '\n')
        wr.close()
    
    @property
    def num_classes(self):
        return self.n_classes


def collate_fn(batch):
    input_ids, labels = [], []
    attention_mask = []
    for item in batch:
        input_ids.append(item['input_ids'])
        attention_mask.append(item['attention_mask'])
        if 'label' in item:
            labels.append(item['label'])
        
    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'labels': torch.tensor(labels)
    }
