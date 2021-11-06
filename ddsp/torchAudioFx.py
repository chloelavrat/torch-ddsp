
## -- library
from enum import auto
import torch
import torch.fft as fft
from torch.functional import norm
import torch.nn as nn
from torch.utils.data.dataset import ConcatDataset

## -- Convolution Reverb
class ConvolReverb(nn.Module):
    def __init__(self, sampling_rate, length, init_wet):
        super().__init__()
        ## -- Parameters
        self.length = length*sampling_rate # in seconds
        self.sampling_rate = sampling_rate
        self.init_wet = init_wet
        ## -- nn.parameters
        self.wet = nn.Parameter(torch.tensor(float(self.init_wet)))
        self.ir = nn.Parameter(
            (torch.rand(self.length) * 2 - 1).unsqueeze(-1)
        )

    def forward(self, x):
        # -- get the impulse response and add wet
        ir = self.ir*torch.sigmoid(self.wet)
        # -- assure the first reverberation
        ir[:, 0] = 1
        # -- zero padding
        ir = nn.functional.pad(ir, (0, 0, 0, x.shape[1] - self.length))
        # -- convolution : ir * x
        x = x.squeeze(-1)
        x = nn.functional.pad(x, (0, x.shape[-1]))
        ir = nn.functional.pad(ir.squeeze(-1), (ir.squeeze(-1).shape[-1], 0))
        print(ir.shape)
        print(x.shape)
        x = fft.irfft(fft.rfft(x) * fft.rfft(ir))
        x = x[..., x.shape[-1] // 2:]
        return x

class Filter(nn.Module):
    def __init__(self, sampling_rate, length):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.length = length

    def forward(self, x, coeff):
        print("---")
        # -- zero padding
        coeff = nn.functional.pad(coeff, (0, 0, 0, x.shape[1] - self.length)).unsqueeze(-1)
        # -- convolution : ir * x
        x = x.squeeze(-1)
        x = nn.functional.pad(x, (0, x.shape[-1]))
        coeff = nn.functional.pad(coeff.squeeze(-1), (coeff.squeeze(-1).shape[-1], 0))
        print(coeff.shape)
        print(x.shape)
        x = fft.irfft(fft.rfft(x) * fft.rfft(coeff))
        x = x[..., x.shape[-1] // 2:]
        return x

if __name__ == "__main__":
    from data_loader import AudioLoader
    import matplotlib.pyplot as plt

    audio_ldr = AudioLoader(
        data_path = "database\\data\\",
        block_size=160,
        sequence_size=100,
        mono=True,
        sampling_rate=16000
    )
    reverb = ConvolReverb(
        sampling_rate=audio_ldr.sampling_rate,
        length=1,
        init_wet=0
    )
    filter = Filter(
        sampling_rate=audio_ldr.sampling_rate,
        length=audio_ldr.block_size
    )

    audio = audio_ldr.__getitem__(204)

    audio_reverb = reverb.forward(audio)

    audio_filtered = filter.forward(
        audio,
        torch.tensor([1,1,1,1,1,0,0,0,0]).unsqueeze(0)
    )



    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('A tale of 3 subplots')

    ax1.plot(audio[0,:])
    ax1.set_ylabel('RawAudio')

    ax2.plot(audio_reverb[0,:].detach().numpy())
    ax2.set_ylabel('audio_reverb')

    import soundfile as sf
    sf.write(
        "database\\data\\audio_data_rev.wav", 
        audio_reverb[0,:].detach().numpy(), 
        audio_ldr.sampling_rate)
    #ax3.plot(audio[0,:])
    ax3.set_ylabel('audio')
    plt.show()