from os import path
import torch
import glob, os
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, dataloader

class AudioLoader(Dataset):
    def __init__(self, data_path, block_size, sequence_size, mono=True, sample_rate=16000):
        super().__init__()
        ## -- Parameters
        self.data_path = data_path
        self.block_size = block_size
        self.sequence_size = sequence_size
        self.mono = mono
        self.sample_rate = sample_rate
        ## -- load files path
        if (os.path.exists(self.data_path)):
            self.files = sorted(glob.glob(self.data_path + '/*'))
        else:
            raise Exception('Unknown raw data path : ' + self.data_path)
        ## -- vars
        self.frame = self.sequence_size * self.block_size
        ## -- slicing audio
        self.slices = self.slicer()
        self.len = self.slices.shape[1]

    def slicer(self):
        print(">>> Loading samples <<<")
        slices = torch.zeros(1,1,self.frame)
        for f in self.files:
            print(".Loading : "+f, end="")
            # Load audio
            wav, sample_rate = self.loadAudioFile(f)
            # crop audio
            cropped_size = int(wav.shape[1]/(self.frame))
            wav = wav[:,:cropped_size*self.frame]
            # slicing audio
            wav = wav.view(1,cropped_size,self.frame)
            slices = torch.cat([slices,wav], dim=1)
            print(" : Done")
        # Remove first "artificial" slice
        slices = slices[:,1:,:]
        return slices

    def loadAudioFile(self, path):
        data, sample_rate  = torchaudio.load(path)
        ## -- resampling
        if sample_rate != self.sample_rate:
            resampler = T.Resample(sample_rate , self.sample_rate, dtype=data.dtype)
            sample_rate = self.sample_rate
            data = resampler(data)
        ## -- stereo to mono
        if self.mono == True:
            data = torch.mean(data, dim=0)
            data = data.unsqueeze(0)
        ## -- done
        return data, sample_rate

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        return self.slices[:,index,:]

if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt

    device_id = torch.cuda.device_count()
    device = torch.cuda.get_device_name(range(device_id))

    data_path = "database\\data\\"

    data_ldr = AudioLoader(
                    data_path, 
                    block_size=160, 
                    sequence_size=100,
                    sample_rate=16000,
                    mono=True)
    