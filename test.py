import os
import random
import argparse
import numpy as np

import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import utils

from conv import *
from utils_data import *
from utils_trainer import *

import evaluate
from transformers import AutoFeatureExtractor, AutoConfig, ASTForAudioClassification
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import datasets
from datasets import Audio

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='FinalData', type=str)
parser.add_argument('-dp', '--data_path', default='./data', type=str)
parser.add_argument('-sp', '--split', default=None, type=str)
parser.add_argument('-m', '--model', default='wav2vec', type=str)
parser.add_argument('-p', '--pretrain', action='store_true')
parser.add_argument('-t', '--transforms', default=None, type=str)
parser.add_argument('-s', '--save_dir', default='./results', type=str)
parser.add_argument('-e', '--epochs', default=50, type=int)
parser.add_argument('-o', '--optim', default='ADAM', type=str)
parser.add_argument('-lr', '--learning_rate', default=1e-04, type=float)
parser.add_argument('-wd', '--weight_decay', default=1e-04, type=float)
parser.add_argument('-vc', '--valid_classes', default='all', help='all to train on all 10 classes otherwise 6', type=str)
parser.add_argument('-lr_sc', '--lr_scheduler', default='Linear', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

if args.data == 'FinalData':
    data_dir = os.path.join(args.data_path, 'FinalData')
    data, (labels, classes, label2id) = get_finaldata(data_dir)
    id2label = dict()
    for i, label in enumerate(classes):
        id2label[str(i)] = label

    max_length = 3
    pid_labels = None
    num_labels = len(classes)

elif args.data == 'violin_data':
    data_dir = args.data_path
    if args.valid_classes == "all":
        data, (labels, classes, label2id), weights = get_data(data_dir)
    else:
        data, (labels, classes, label2id), weights = get_data(data_dir, valid_classes=['detache', 'legato', 'martele', 'spiccato', 'flying staccato', 'tremolo'])
    id2label = dict()
    for i, label in enumerate(classes):
        id2label[str(i)] = label

    max_length = 5
    pid_labels = None
    num_labels = len(classes)    

else:
    data_dir = os.path.join(args.data_path, 'dataset-master-violin')
    data, (labels, classes, label2id), (pid_labels, pid_classes, pid2id) = get_dataset_violin(data_dir)
    id2label = dict()
    for i, label in enumerate(classes):
        id2label[str(i)] = label

    max_length = 5
    pid_labels = None
    num_classes = len(classes)

violindataset = custom_train_test_split(data, labels, pid=pid_labels)

violin_datasets = datasets.Dataset.from_dict({'audio': violindataset['test'][0], 'label':violindataset['test'][1]})

batch_size = len(violin_datasets) 

violin_datasets = violin_datasets.class_encode_column("label")

if args.model == 'wave2vec':
    model_str = "facebook/wav2vec2-base"
elif args.model == 'hubert':
    model_str = "ntu-spml/distilhubert"
else:
    model_str = "MIT/ast-finetuned-audioset-10-10-0.4593"


feature_extractor = AutoFeatureExtractor.from_pretrained(args.save_dir)
violin_datasets = violin_datasets.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
    audio_arrays, sampling_rate=feature_extractor.sampling_rate, return_tensors='pt', padding=True, truncation=True, max_length=max_length*feature_extractor.sampling_rate)
    
    return inputs

violin_datasets = violin_datasets.map(preprocess_function, remove_columns="audio", batched=True)

model = AutoModelForAudioClassification.from_pretrained(args.save_dir, num_labels=num_labels, label2id=label2id, id2label=id2label)

training_args = TrainingArguments(
    output_dir=os.path.join(args.save_dir, args.model),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=args.epochs,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=5,
    dataloader_num_workers = 1 if device == "cuda" else 0,
    metric_for_best_model="accuracy",
    )

trainer = Trainer(
    model=model,
    train_dataset=violin_datasets,
    eval_dataset=violin_datasets,
    compute_metrics=compute_metrics,
)


trainer_evaluation = trainer.predict(violin_datasets)

predictions = trainer_evaluation.predictions
labels = trainer_evaluation.label_ids
preds = np.argmax(predictions, axis=1)

raw_predictions = {'preds': preds.tolist(), 'labels':labels.tolist()}

with open(os.path.join(args.save_dir, 'raw_predictions.json'), 'w') as f:
    json.dump(raw_predictions, f)

f1_metric = evaluate.load("f1")
f1_results = f1_metric.compute(predictions=preds, references=labels, average='weighted')

accuracy_results = accuracy.compute(predictions=preds, references=labels)

cm = confusion_matrix(y_true=labels , y_pred=preds)

# plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=id2label)
disp.plot()
plt.savefig(os.path.join(args.save_dir, 'test_confusion_matrix.png'))

results={'f1': f1_results['f1'], 'accuracy': accuracy_results['accuracy']}
print(results)

with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
    json.dump(results, f)

