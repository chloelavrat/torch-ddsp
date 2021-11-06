import numpy as np
import librosa
import torch
from torch._C import device 
import torchcrepe
import torch.fft as fft

def extract_loudness(audio, sampling_rate, block_size, n_fft=2048,lr=1e-3):
    S = torch.stft(
        input=audio,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
        return_complex=True
    )
    S = torch.log10(torch.abs(S)+lr)
    # Perceptual weighting.
    f = librosa.A_weighting(librosa.fft_frequencies(sampling_rate,n_fft))
    f = torch.tensor(f, device=audio.device).unsqueeze(0)
    S = S + f.reshape(-1, 1)
    S = torch.mean(S, 0)[..., :-1]
    return S

def extract_pitch(audio, sampling_rate, block_size, threshold_conf=0.15, model="tiny", lr=1e-4):
    device = audio.device
    pitch, periodicity = torchcrepe.predict(audio,
                           sampling_rate,
                           block_size,
                           0,
                           sampling_rate/2,
                           model,
                          # batch_size=batch_size,
                           decoder=torchcrepe.decode.viterbi,
                           return_periodicity=True,
                           device=device)
    # We'll use a 15 millisecond window assuming a hop length of 5 milliseconds
    win_length = 2

    # Median filter noisy confidence value
    periodicity = torchcrepe.filter.median(periodicity, win_length)

    # Remove extra value added by crepe
    pitch = pitch[:, :-1]
    periodicity = periodicity[:, :-1]

    # Remove inharmonic regions
    pitch = torch.where(
        periodicity < threshold_conf, 
        torch.tensor(lr ,device=audio.device), 
        pitch
    )

    print(pitch.shape)
    # Optionally smooth pitch to remove quantization artifacts
    pitch = torchcrepe.filter.mean(pitch, win_length)
    return pitch

if __name__ == "__main__":
    from data_loader import AudioLoader
    import matplotlib.pyplot as plt

    audio_ldr = AudioLoader(
        data_path = "database\\data\\",
        block_size=160*3,
        sequence_size=100,
        mono=True,
        sampling_rate=16000
    )
    # -- get audio slice
    item1 = audio_ldr.__getitem__(203)
    # -- apply : extract_pitch on it
    pitch = extract_pitch(
        item1, 
        sampling_rate=audio_ldr.sampling_rate,
        block_size=audio_ldr.block_size
    )
    # -- apply : extract_loudness on it
    loudness = extract_loudness(
        item1, 
        sampling_rate=audio_ldr.sampling_rate, 
        block_size=audio_ldr.block_size
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('A tale of 3 subplots')

    ax1.plot(item1[0,:])
    ax1.set_ylabel('RawAudio')

    ax2.plot(pitch[0,:])
    ax2.set_ylabel('pitch')

    ax3.plot(loudness[0,:])
    ax3.set_ylabel('loudness')
    plt.show()
    