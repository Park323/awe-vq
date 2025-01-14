"""
Functions for extracting filterbank and MFCC features.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from os import path
from tqdm import tqdm
import glob
import librosa
import numpy as np
import scipy.io.wavfile as wav

from torch import tensor
from selfsup import Wav2VecExtractor, HuBERTExtractor


def extract_fbank_dir(dir=None, paths=None):
    """
    Extract filterbanks for all audio files in `dir` and return a dictionary.

    Each dictionary key will be the filename of the associated audio file
    without the extension. Mel-scale log filterbanks are extracted.
    """
    assert paths is not None or dir is not None
    if paths is None: 
        paths = sorted(glob.glob(path.join(dir, '*.wav')))
    
    feat_dict = {}
    for wav_fn in tqdm(paths):
        signal, sample_rate = librosa.core.load(wav_fn, sr=None)
        signal = preemphasis(signal, coeff=0.97)
        fbank = np.log(librosa.feature.melspectrogram(
            signal, sr=sample_rate, n_mels=40,
            n_fft=int(np.floor(0.025*sample_rate)),
            hop_length=int(np.floor(0.01*sample_rate)), fmin=64, fmax=8000,
            ))
        # from python_speech_features import logfbank
        # samplerate, signal = wav.read(wav_fn)
        # fbanks = logfbank(
        #     signal, samplerate=samplerate, winlen=0.025, winstep=0.01,
        #     nfilt=45, nfft=2048, lowfreq=0, highfreq=None, preemph=0,
        #     winfunc=np.hamming
        #     )
        key = path.splitext(path.split(wav_fn)[-1])[0]
        feat_dict[key] = fbank.T
    return feat_dict


def extract_mfcc_dir(dir=None, paths=None):
    """
    Extract MFCCs for all audio files in `dir` and return a dictionary.

    Each dictionary key will be the filename of the associated audio file
    without the extension. Deltas and double deltas are also extracted.
    """
    assert paths is not None or dir is not None
    if paths is None: 
        paths = sorted(glob.glob(path.join(dir, '*.wav')))
    
    feat_dict = {}
    for wav_fn in tqdm(paths):
        signal, sample_rate = librosa.core.load(wav_fn, sr=None)
        signal = preemphasis(signal, coeff=0.97)
        mfcc = librosa.feature.mfcc(
            signal, sr=sample_rate, n_mfcc=13, n_mels=24,  #dct_type=3,
            n_fft=int(np.floor(0.025*sample_rate)),
            hop_length=int(np.floor(0.01*sample_rate)), fmin=64, fmax=8000,
            #htk=True
            )
        # mfcc = librosa.feature.mfcc(
        #     signal, sr=sample_rate, n_mfcc=13,
        #     n_fft=int(np.floor(0.025*sample_rate)),
        #     hop_length=int(np.floor(0.01*sample_rate))
        #     )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
        # from python_speech_features import delta
        # from python_speech_features import mfcc
        # samplerate, signal = wav.read(wav_fn)
        # mfccs = mfcc(
        #     signal, samplerate=samplerate, winlen=0.025, winstep=0.01,
        #     numcep=13, nfilt=24, nfft=None, lowfreq=0, highfreq=None,
        #     preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming
        #     )
        # d_mfccs = delta(mfccs, 2)
        # dd_mfccs = delta(d_mfccs, 2)
        key = path.splitext(path.split(wav_fn)[-1])[0]
        feat_dict[key] = np.hstack([mfcc.T, mfcc_delta.T, mfcc_delta_delta.T])
        # import matplotlib.pyplot as plt
        # plt.imshow(feat_dict[key][2000:2200,:])
        # plt.show()
        # assert False
    return feat_dict


def extract_w2v_dir(
    dir=None, paths=None, vad_dict=None, postprocess=None, 
    key_generator=None, device='cpu'):
    """
    Extract Wav2vec2.0 feature for all audio files in `dir` and return a dictionary.

    Each dictionary key will be the filename of the associated audio file
    without the extension. Variable-lengthed wav2vec features are extracted.
    
    [Options]
    model_scale
    model_layer
    model_device
    postprocess
    """
    assert paths is not None or dir is not None
    if paths is None: 
        paths = sorted(glob.glob(path.join(dir, '*.wav')))
    
    wav2vec = Wav2VecExtractor(model_scale='large', device=device)
    wav2vec.eval()
    feat_dict = {}
    vq_dict = {}
    for wav_fn in tqdm(paths):
        signal, sample_rate = librosa.core.load(wav_fn, sr=None)
        if vad_dict:
            utt_key = key_generator(wav_fn)
            if utt_key not in vad_dict:
                print('Warning: missing VAD for utterance', utt_key)
            else:
                for start, end in vad_dict[utt_key]:
                    start_ = int(round(start * sample_rate))
                    end_ = int(round(end * sample_rate)) + 1
                    segment = signal[start_:end_]
                    if segment.shape[0] < 10:
                        import pdb; pdb.set_trace()
                    segment = tensor(segment).unsqueeze(0)
                    feature, vq_idx = wav2vec.extract(inputs=segment, sr=sample_rate)
                    if postprocess is not None:
                        feature = postprocess(feature)
                    sid = int(round(start * 100))
                    eid = int(round(end * 100)) + 1
                    segment_key = utt_key + '_{:06d}-{:06d}'.format(sid, eid)
                    feat_dict[segment_key] = feature.cpu().detach().numpy()
                    vq_dict[segment_key] = vq_idx.cpu().detach().numpy()
        else:
            signal = tensor(signal).unsqueeze(0)
            feature, vq_idx = wav2vec.extract(inputs=signal, sr=sample_rate)
            if postprocess is not None:
                feature = postprocess(feature)
            if key_generator:
                key = key_generator(wav_fn)
            else:
                key = path.splitext(path.split(wav_fn)[-1])[0]
            feat_dict[key] = feature.cpu().detach().numpy()
            vq_dict[key] = vq_idx.cpu().detach().numpy()
    
    print('Total', len(feat_dict), 'features are extracted.')
    
    return (feat_dict, vq_dict)


def extract_hb_dir(
    dir=None, paths=None, vad_dict=None, postprocess=None, 
    key_generator=None, device='cpu'):
    """
    Extract HuBERT feature for all audio files in `dir` and return a dictionary.

    Each dictionary key will be the filename of the associated audio file
    without the extension. Variable-lengthed HuBERT features are extracted.
    """
    assert paths is not None or dir is not None
    if paths is None: 
        paths = sorted(glob.glob(path.join(dir, '*.wav')))
    
    hubert = HuBERTExtractor(model_scale='large', device=device)
    hubert.eval()
    feat_dict = {}
    vq_dict = {}
    for wav_fn in tqdm(paths):
        signal, sample_rate = librosa.core.load(wav_fn, sr=None)
        if vad_dict:
            utt_key = key_generator(wav_fn)
            if utt_key not in vad_dict:
                print('Warning: missing VAD for utterance', utt_key)
            else:
                for start, end in vad_dict[utt_key]:
                    start_ = int(round(start * sample_rate))
                    end_ = int(round(end * sample_rate)) + 1
                    segment = signal[start_:end_]
                    segment = tensor(segment).unsqueeze(0)
                    feature = hubert.extract(inputs=segment, sr=sample_rate)
                    if postprocess is not None:
                        feature = postprocess(feature)
                    sid = int(round(start * 100))
                    eid = int(round(end * 100)) + 1
                    segment_key = utt_key + '_{:06d}-{:06d}'.format(sid, eid)
                    feat_dict[segment_key] = feature.cpu().detach().numpy()
        else:
            signal = tensor(signal).unsqueeze(0)
            feature = hubert.extract(inputs=signal, sr=sample_rate)
            if postprocess is not None:
                feature = postprocess(feature)
            if key_generator:
                key = key_generator(wav_fn)
            else:
                key = path.splitext(path.split(wav_fn)[-1])[0]
            feat_dict[key] = feature.cpu().detach().numpy()

    print('Total', len(feat_dict), 'features are extracted.')
    return feat_dict


def extract_vad(feat_dict, vad_dict):
    """
    Remove silence based on voice activity detection (VAD).

    The `vad_dict` should have the same keys as `feat_dict` with the active
    speech regions given as lists of tuples of (start, end) frame, with the end
    excluded.
    """
    output_dict = {}
    for utt_key in tqdm(sorted(feat_dict)):
        if utt_key not in vad_dict:
            print("Warning: missing VAD for utterance", utt_key)
            continue
        for (start, end) in vad_dict[utt_key]:
            segment_key = utt_key + "_{:06d}-{:06d}".format(start, end)
            output_dict[segment_key] = feat_dict[utt_key][start:end, :]
            assert output_dict[segment_key].shape[0] != 0, \
                f"Feature from {utt_key} has a zero length!"
    return output_dict


def speaker_mvn(feat_dict):
    """
    Perform per-speaker mean and variance normalisation.

    It is assumed that each of the keys in `feat_dict` starts with a speaker
    identifier followed by an underscore.
    """

    speakers = set([key.split("_")[0] for key in feat_dict])

    # Separate features per speaker
    speaker_features = {}
    for utt_key in sorted(feat_dict):
        speaker = utt_key.split("_")[0]
        if speaker not in speaker_features:
            speaker_features[speaker] = []
        speaker_features[speaker].append(feat_dict[utt_key])

    # Determine means and variances per speaker
    speaker_mean = {}
    speaker_std = {}
    for speaker in speakers:
        features = np.vstack(speaker_features[speaker])
        speaker_mean[speaker] = np.mean(features, axis=0)
        speaker_std[speaker] = np.std(features, axis=0)

    # Normalise per speaker
    output_dict = {}
    for utt_key in tqdm(sorted(feat_dict)):
        speaker = utt_key.split("_")[0]
        output_dict[utt_key] = (
            (feat_dict[utt_key] - speaker_mean[speaker]) / 
            speaker_std[speaker]
            )

    return output_dict


def preemphasis(signal, coeff=0.97):
    """Perform preemphasis on the input `signal`."""    
    return np.append(signal[0], signal[1:] - coeff*signal[:-1])
