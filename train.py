import os
import random
import argparse
import numpy as np
import json

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
parser.add_argument('-d', '--data', default='violin_data', type=str)
parser.add_argument('-dp', '--data_path', default='violin_data/bowingdirection_processed_wav', type=str)
parser.add_argument('-sp', '--split', default=None, type=str)
parser.add_argument('-m', '--model', default='wav2vec', type=str)
parser.add_argument('-p', '--pretrain', action='store_true')
parser.add_argument('-t', '--transforms', default=None, type=str)
parser.add_argument('-s', '--save_dir', default='./results_violin_10', type=str)
parser.add_argument('-b', '--batch_size', default=32, type=int)
parser.add_argument('-e', '--epochs', default=50, type=int)
parser.add_argument('-o', '--optim', default='ADAM', type=str)
parser.add_argument('-lr', '--learning_rate', default=1e-04, type=float)
parser.add_argument('-wd', '--weight_decay', default=1e-04, type=float)
parser.add_argument('-w', '--weights', default='equal', help='One of: equal, 10, scale', type=str)
parser.add_argument('-vc', '--valid_classes', default='all', help='all to train on all 10 classes otherwise 6', type=str)
parser.add_argument('-lr_sc', '--lr_scheduler', default='Linear', type=str)
args = parser.parse_args()

seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

save_name = f"{args.model}_d{args.data}_b{args.batch_size}_o{args.optim}_lr{args.learning_rate}_w{args.weights}_vc{args.valid_classes}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#if args.transforms == 'spectrogram':
#    transforms = torch.nn.Sequential(
#        torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2),
#        torchaudio.transforms.TimeStretch(n_freq=n_fft, fixed_rate=True),
#        torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param),
#        torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param),
#        torchvision.transforms.Resize(size=(size,size)),
#        normalize(),
#        )
#elif args.transforms == 'mel_spectrogram':
#    transforms = torch.nn.Sequential(
#        torchaudio.transforms.MelSpectrogram(sample_rate=target_sample_rate,n_fft=n_fft),
#        torchaudio.transforms.TimeStretch(n_freq = n_fft, fixed_rate=True),
#        torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param),
#        torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param),
#        torchvision.transforms.Resize(size=(size,size)),
#        normalize(),
#        )


if args.data == 'FinalData':
    data_dir = os.path.join(args.data_path, 'FinalData')
    data, (labels, classes, label2id) = get_finaldata(data_dir)
    id2label = dict()
    for i, label in enumerate(classes):
        id2label[str(i)] = label

    max_length = 5
    pid_labels = None
    num_labels = len(classes)

elif args.data == 'violin_data':
    #data_dir = os.path.join(args.data_path, 'FinalData')
    data_dir = args.data_path
    if args.valid_classes == "all":
        data, (labels, classes, label2id), weights = get_data(data_dir)
    else:
        data, (labels, classes, label2id), weights = get_data(data_dir, valid_classes=['detache', 'legato', 'martele', 'spiccato', 'flying staccato', 'tremolo'])
    id2label = dict()
    for i, label in enumerate(classes):
        id2label[str(i)] = label

    if args.weights == "equal":
        weights = [1.0 for x in weights]
    elif args.weights == "10":
        weights = [10.0 if x >100 else 1.0 for x in weights]
    elif weights == "scale":
         weights = weights
    
    max_length = 3
    pid_labels = None
    num_labels = len(classes)

else:
    data_dir = os.path.join(args.data_path, 'dataset-master-violin')
    data, (labels, classes, label2id), (pid_labels, pid_classes, pid2id) = get_dataset_violin(data_dir)
    id2label = dict()
    for i, label in enumerate(classes):
        id2label[str(i)] = label

    max_length = 10
    pid_labels = pid_labels if args.split is not None else None 
    num_classes = len(classes)

violindataset = custom_train_test_split(data, labels, pid=pid_labels)

print(f"There are {len(violindataset['train'][0])} samples in the train dataset.")
print(f"There are {len(violindataset['val'][0])} samples in the val dataset.")
print(f"There are {len(violindataset['test'][0])} samples in the test dataset.")

if args.model == 'conv':


    target_sample_rate = 8000

    transforms = torch.nn.Sequential(
    #RandomSpeedChange(target_sample_rate),
    RandomClip(target_sample_rate * (max_length)),)
    
    violin_datasets = {x: ViolinDataset(violindataset[x],
                            transforms,
                            target_sample_rate,
                            max_length,
                            ) for x in ['train', 'val', 'test']}


    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    data_loaders = {x: DataLoader(
                    violin_datasets[x],
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    ) for x in ['train', 'val', 'test']}

    model = conv_model(num_classes=num_labels)    

    trainer_args = {'model': model, 'save_dir': os.path.join(args.save_dir, args.model), 'save_name': save_name, 'optimizer':args.optim, 'lr': args.learning_rate, 'weight_decay': args.weight_decay, 'adam_beta1': 0.9, 'adam_beta2': 0.999, 'momentum':0.9, 'epochs': args.epochs, 'lr_scheduler': args.lr_scheduler, 'device':device}

    class WeightedPytorchTrainer(PytorchTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get('logits')
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda())
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    Trainer = PytorchTrainer(**trainer_args)
    Trainer.train(data_loaders['train'], data_loaders['val'])  

elif args.model in ['wav2vec', 'distilhubert', 'ast']:

    violindataset = {x : {'audio': violindataset[x][0], 'label':violindataset[x][1]} for x in ['train', 'val', 'test']}
    
    violin_datasets = datasets.DatasetDict({x: datasets.Dataset.from_dict(violindataset[x]).cast_column("audio", Audio()) for x in ['train', 'val', 'test']})

    violin_datasets = violin_datasets.class_encode_column("label")

    if args.model == 'wav2vec':    
        model_str = "facebook/wav2vec2-base"
        model = AutoModelForAudioClassification.from_pretrained(model_str, num_labels=num_labels, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
    elif args.model == 'hubert':
        model_str = "ntu-spml/distilhubert"
    else:
        model_str = "MIT/ast-finetuned-audioset-10-10-0.4593"
        model = AutoModelForAudioClassification.from_pretrained(model_str, num_labels=num_labels, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
        
        
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_str)
    violin_datasets = violin_datasets.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, return_tensors='pt', padding=True, truncation=True, max_length=max_length*feature_extractor.sampling_rate 
        )
        return inputs

    violin_datasets = violin_datasets.map(preprocess_function, remove_columns="audio", batched=True)

    if not args.pretrain: 
        config = AutoConfig.from_pretrained(model_str, num_labels = num_labels)
        model = AutoModelForAudioClassification.from_config(config)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get('logits')
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda())
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
    output_dir=os.path.join(args.save_dir, args.model),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay, 
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=5,
    dataloader_num_workers = 1 if device == "cuda" else 0,
    metric_for_best_model="accuracy",
    )
    
    trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=violin_datasets["train"],
    eval_dataset=violin_datasets["val"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,)

    trainer.train()

trainer_evaluation = trainer.predict(violin_datasets['test'])

predictions = trainer_evaluation.predictions
labels = trainer_evaluation.label_ids
preds = np.argmax(predictions, axis=1)

raw_predictions = {'preds': preds.tolist(), 'labels':labels.tolist()}

with open(os.path.join(os.path.join(args.save_dir, args.model), 'raw_predictions.json'), 'w') as f:
    json.dump(raw_predictions, f)

f1_metric = evaluate.load("f1")
f1_results = f1_metric.compute(predictions=preds, references=labels, average='weighted')

accuracy_results = accuracy.compute(predictions=preds, references=labels)

cm = confusion_matrix(y_true=labels , y_pred=preds)

# plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=id2label)
disp.plot()
plt.savefig(os.path.join(os.path.join(args.save_dir, args.model), 'test_confusion_matrix.png'))

results={'f1': f1_results['f1'], 'accuracy': accuracy_results['accuracy']}
print(results)

with open(os.path.join(os.path.join(args.save_dir, args.model), 'results.json'), 'w') as f:
    json.dump(results, f)
