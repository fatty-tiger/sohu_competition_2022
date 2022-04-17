import sys
import os
import logging
import collections
import time
import numpy as np
import torch

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from reader import sohunlp_dataset as dataset


def compute_metrics(output):
    labels = output.label_ids
    preds = output.predictions
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=None).tolist()
    f1_detail = ', '.join([str(x) for x in f1])
    f1_macro = f1_score(labels, preds, average='macro')
    return {
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_detail': f1_detail
    }


@torch.no_grad()
def inference(model, device, test_dataloader):
    t0 = time.time()
    model.eval()
    total_preds = []
    total_labels = []
    for step, batch in enumerate(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, return_dict=True)
        preds = torch.argmax(output['logits'], dim=1)
        total_preds.extend(preds.reshape(-1).tolist())
        total_labels.extend(batch['labels'].reshape(-1).tolist())
        time_elapsed = int(time.time() - t0)
    logging.info("Inference Time Elapsed': %s" % time_elapsed)
    # 两种方法来给 namedtuple 定义方法名
    ModelOutput = collections.namedtuple('ModelOutput', ['label_ids', 'predictions'])
    ret = ModelOutput(
            np.array(total_labels),
            np.array(total_preds)
        )
    return ret


def train_and_save():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    max_len = 512
    use_distribute = True
    training_args = TrainingArguments(
        output_dir='./output2',          # output directory
        learning_rate=5e-5,
        num_train_epochs=1,              # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=1,   # batch size for evaluation
        dataloader_num_workers=4,
        logging_dir='./logs',            # directory for storing logs
        logging_steps=50,
        log_level='info',
        do_train=True,
        do_eval=False,
        do_predict=False,
        no_cuda=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        label_names=["labels"],
        disable_tqdm=True,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        gradient_accumulation_steps=4,
    )
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    pretrained_dir = "/home/work/jiangjie04/pretrained/huggingface/bert-base-chinese"
    train_file = "../data/sohunlp/train_new.json"
    valid_file = "../data/sohunlp/valid_new_sample.json"
    test_file = "../data/sohunlp/test_new_sample.json"
    my_tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
    train_dataset = dataset.MyDataset(train_file, max_len, my_tokenizer)
    n_classes = train_dataset.num_classes
    print("n_classes=", n_classes)
    print("train_samples=", len(train_dataset))

    model = BertForSequenceClassification.from_pretrained(
            pretrained_dir,
            num_labels=n_classes,
            problem_type="single_label_classification")
    model.to(device)
    if use_distribute:
        device_ids = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        data_collator=dataset.collate_fn,
        train_dataset=train_dataset,         # training dataset
    )
    train_out = trainer.train()


def keep_training():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    max_len = 512
    do_eval = False
    do_predict = False
    use_distribute = True
    # lr_scheduler_type=
    # warmup_ratio
    training_args = TrainingArguments(
        output_dir='./output2',          # output directory
        learning_rate=5e-5,
        num_train_epochs=1,              # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=1,   # batch size for evaluation
        logging_dir='./logs',            # directory for storing logs
        logging_steps=50,
        do_train=True,
        do_eval=do_eval,
        do_predict=do_predict,
        no_cuda=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        label_names=["labels"],
        disable_tqdm=True,
        gradient_accumulation_steps=4,
    )
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    pretrained_dir = "/home/work/jiangjie04/pretrained/huggingface/bert-base-chinese"
    train_file = "../data/sohunlp/train_new.json"
    valid_file = "../data/sohunlp/valid_new_sample.json"
    test_file = "../data/sohunlp/test_new_sample.json"

    my_tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
    train_dataset = dataset.MyDataset(train_file, max_len, my_tokenizer)
    n_classes = train_dataset.num_classes
    print("n_classes=", n_classes)
    print("train_samples=", len(train_dataset))
    val_dataset = None
    test_dataset = None
    config = AutoConfig.from_pretrained(
                pretrained_dir,
                num_labels=n_classes,
                problem_type="single_label_classification")
    model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)
    if use_distribute:
        device_ids = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        data_collator=dataset.collate_fn,
        train_dataset=train_dataset,         # training dataset
    )
    train_out = trainer.train("./output/checkpoint-2571")


def load_and_evaluate():
    checkpoint_path = "./output/checkpoint-2571"
    max_len = 512
    use_distribute = True
    batch_size = 384
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    if use_distribute:
        device_ids = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    pretrained_dir = "/home/work/jiangjie04/pretrained/huggingface/bert-base-chinese"
    my_tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
    test_dataset = dataset.MyDataset('../data/sohunlp/valid_new.json', max_len, my_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size, collate_fn=dataset.collate_fn, num_workers=4)
    print("test_samples=", len(test_dataset))
    output = inference(model, device, test_dataloader)
    metrics = compute_metrics(output)
    print(metrics)


def load_and_predict():
    checkpoint_path = "./output/checkpoint-2571"
    max_len = 512
    use_distribute = True
    batch_size = 384
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    if use_distribute:
        device_ids = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    pretrained_dir = "/home/work/jiangjie04/pretrained/huggingface/bert-base-chinese"
    my_tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
    test_dataset = dataset.MyDataset('../data/sohunlp/test_new.json', max_len, my_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size, collate_fn=dataset.collate_fn, num_workers=4)
    print("test_samples=", len(test_dataset))
    output = inference(model, device, test_dataloader)
    preds = [int(x) for x in output.predictions]
    test_dataset.gen_submission(preds, './output/result_041501.tsv')


if __name__ == '__main__':
    train_and_save()
    #load_and_predict()

