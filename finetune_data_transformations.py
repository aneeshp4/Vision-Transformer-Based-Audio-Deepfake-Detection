import torchaudio.transforms as T
import random

class DataAugmentation:
    def __init__(self, sample_rate=16000, num_mel_bins=128):
        self.transforms = T.AugmentMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=num_mel_bins,
            mel_transforms=[
                T.FrequencyMasking(freq_mask_param=15),
                T.TimeMasking(time_mask_param=35)
            ]
        )
    
    def __call__(self, fbank):
        # Apply transforms with a probability
        if random.random() < 0.5:
            return self.transforms(fbank)
        return fbank
