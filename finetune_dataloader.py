import torchaudio
import os
import csv
import json

from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, dataset_path, sample_rate=16000):
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        
        # Load the JSON file containing the dataset information
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Get the path and label for the current index
        file_info = self.data[index]
        file_path = os.path.join(self.dataset_path, file_info['wav'])
        
        # Load the audio file
        waveform, sr = torchaudio.load(file_path)
        
        # Resample waveform if its sample rate differs from the target sample rate
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        
        # Convert waveform to fbank features
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, 
            sample_frequency=self.sample_rate, 
            num_mel_bins=128,  # Adjust the number of mel bins as necessary
            frame_shift=10,
            frame_length=25,
            use_energy=False
        )
        
        # Normalize fbank features
        fbank = (fbank - fbank.mean()) / fbank.std()

        # fbank = fbank.unsqueeze(0)  # Shape becomes [1, frames, features]
        
        # Map labels to integers if required, or use strings directly
        label_map = {'FAKE': 0, 'REAL': 1}  # Adjust according to actual labels
        label = label_map[file_info['label']]
        
        return fbank, label
