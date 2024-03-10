import csv
import json
import torchaudio
import os

from torch.utils.data import Dataset

 
def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, dataset_path, sample_rate=16000, label_csv=None, transform=None):

        self.dataset_path = dataset_path
        
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']

        self.sample_rate = sample_rate
        self.transform=transform
        



    def __getitem__(self, index):

        file_path = os.path.join(self.dataset_path, self.data[index]['wav'])
        waveform, sr = torchaudio.load(file_path)
        resample_fn = torchaudio.transforms.Resample(sr, self.sample_rate)
        waveform = resample_fn(waveform)
        
        if waveform.shape[1] == 0:
            return (None, None, None), None
        
        if self.transform is not None:
            waveform = self.transform(waveform)
            
        target = -1
            
        return waveform, target
    

    def __len__(self):
        return len(self.data)
