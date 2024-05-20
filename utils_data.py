import os
import random
import torch
from torch.utils.data import Dataset
import torchaudio
import librosa


def get_data(data_dir, valid_classes=['colle','detache','flying staccato','martele','legato','ricochet','sautille','spiccato','staccato','tremolo']):
    wav_files = []
    weights = {}
    labels = []
    valid_classes = sorted(valid_classes)
    classes = sorted(entry.name for entry in os.scandir(data_dir) if (entry.is_dir() and entry.name in valid_classes))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    for c in classes:
        listwav = os.listdir(os.path.join(data_dir, c))
        weights[c] = len(listwav)
        for file in listwav:
            if file.endswith('.wav'):
                y, sr = librosa.load(os.path.join(data_dir, c, file))
                if y.shape[0] > 0:
                    wav_files.append(os.path.join(data_dir, c, file))
                    labels.append(class_to_idx[c])
    
    w = [value for _, value in weights.items()]
    w = [sum(w)/x for x in w]
    return wav_files, (labels, classes, class_to_idx), w


def get_dataset_violin(data_dir):
    wav_files = []
    labels_p = []
    labels = []
    classes = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    bow_strokes = []
    bow_strokes_to_idx = {}
    for c in classes:
        for file in os.listdir(os.path.join(data_dir, c)):
            if file.endswith('.wav'):
                wav_files.append(os.path.join(data_dir, c, file))
                labels_p.append(class_to_idx[c])
                bowstroke = file.split('_')[-3]
                if bowstroke not in bow_strokes:
                    bow_strokes_to_idx[bowstroke] = len(bow_strokes)
                    bow_strokes.append(bowstroke)
                labels.append(bow_strokes_to_idx[bowstroke])

    return wav_files, (labels, bow_strokes, bow_strokes_to_idx), (labels_p, classes, class_to_idx)

def custom_train_test_split(data, labels, pid=None):
    assert len(data) == len(labels)
    N = len(data)
    datasets = {}
    if pid is not None:
        unique_pids = set(pid)
        train_data = [data[i] for i in range(N) if pid[i] not in unique_pids[-2:]]
        train_labels = [labels[i] for i in range(N) if pid[i] not in unique_pids[-2:]]
        val_data = [data[i] for i in range(N) if pid[i] == unique_pids[-2]]
        val_labels = [labels[i] for i in range(N) if pid[i] == unique_pids[-2]]
        test_data = [data[i] for i in range(N) if pid[i] == unique_pids[-1]]
        test_labels = [labels[i] for i in range(N) if pid[i] == unique_pids[-1]]

    else:
        # each class represented equally in train, val and test
        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        test_data = []
        test_labels = []
        unique_labels = set(labels)
        for i in unique_labels:
            data_i = [data[j] for j in range(N) if labels[j]==i]
            M = len(data_i)
            random.shuffle(data_i)
            train_data += data_i[:int(0.8*M)]
            train_labels += [i for j in range(int(0.8*M))]
            val_data += data_i[int(0.8*M):int(0.9*M)]
            val_labels += [i for j in range(int(0.8*M),int(0.9*M),1)]
            test_data += data_i[int(0.9*M):]
            test_labels += [i for j in range(int(0.9*M),M,1)]

    datasets['train'] = [train_data, train_labels]
    datasets['val'] = [val_data, val_labels]
    datasets['test'] = [test_data, test_labels]

    return datasets

# audio transforms

class normalize(torch.nn.Module):

    def forward(self, sample):
        sample = ((sample - torch.min(sample)) / torch.max(sample)) * 255
        return sample

class RandomSpeedChange(torch.nn.Module):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        speed_factor = random.choice([0.9, 1.0, 1.1])
        if speed_factor == 1.0: # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects)
        return transformed_audio

class RandomClip(torch.nn.Module):
    def __init__(self, clip_length):
        self.clip_length = clip_length

    def __call__(self, audio_data):
        audio_length = audio_data.shape[1]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length-self.clip_length)
            audio_data = audio_data[:,offset:(offset+self.clip_length)]

        return audio_data # remove silences at the beggining/end


class ViolinDataset(Dataset):

    def __init__(self,
                 data,
                 transforms=None,
                 target_sample_rate=None,
                 max_length = 3,
                 ):
        data, labels = data
        self.labels = torch.tensor(labels)
        self.data = data
        self.transforms = transforms
        self.target_sample_rate = target_sample_rate
        self.max_length = max_length
        self.max_samples = self.max_length * self.target_sample_rate

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        y, sr = torchaudio.load(self.data[index])
        y = self.stereo_to_mono(y)
        y = torchaudio.transforms.Resample(sr, self.target_sample_rate)(y) if (self.target_sample_rate is not None and sr!= self.target_sample_rate) else y
        y = self.truncate(y)
        y = self.pad(y)
        if self.transforms:
            y = self.transforms(y)

        return y, self.labels[index]

    def truncate(self, y):
        if y.shape[1] > self.max_samples:
            y = y[:, :self.max_samples]
        return y

    def pad(self, y):
        length_y = y.shape[1]
        if length_y < self.max_samples:
            pad_length = self.max_samples - length_y
            p1d = (0, pad_length)
            y = torch.nn.functional.pad(y, p1d)
        return y

    def stereo_to_mono(self, y):
        if y.shape[0] > 1:
            y = torch.mean(y, dim=0, keepdim=True)
        return y
