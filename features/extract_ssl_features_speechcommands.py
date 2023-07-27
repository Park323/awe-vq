#!/usr/bin/env python

"""
Extract Self-supervised features for the SpeechCommands V2 dataset.

Author: Jeongkyun Park
Contact: park32323@gmail.com
Date: 2023
"""

from datetime import datetime
from os import path
from tqdm import tqdm
import numpy as np
import os
import glob
import sys

sys.path.append("..")

from paths import speechcommands_datadir
import features
import utils


def extract_features_for_subset(subset, feat_type, output_fn, device='cpu'):
    """
    Extract specified features.

    The `feat_type` parameter can be "w2v" or "hb".
    """
    
    # File list for subset
    file_fn = path.join(
        "..", "data", "speechcommands_" + subset + ".wrd"
        )
    print("Reading:", file_fn)
    filenames = set()
    with open(file_fn) as f:
        for line in f:
            path_, start, end, cls = line.strip().split()
            _, folder, basefile = path_.split('-')
            filenames.add(f"{folder}/{basefile}.wav")
    
    def keygen(fn):
        return path.splitext("-".join(["sc"]+fn.split(os.sep)[-2:]))[0]
    
    # Raw features
    feat_dict = {}
    vq_dict = {}
    vq_dict_wavkey = None
    paths = sorted([path.join(speechcommands_datadir, filename) for filename in filenames])
    postprocess = utils.get_postprocessor(feat_type)
    if "w2v" in feat_type:
        feat_dict_wavkey, vq_dict_wavkey = features.extract_w2v_dir(
            paths=paths,
            postprocess=postprocess,
            key_generator=keygen,
            device=device,
            )
    elif "hb" in feat_type:
        feat_dict_wavkey = features.extract_hb_dir(
            paths=paths,
            postprocess=postprocess,
            key_generator=keygen,
            device=device,
            )
    else:
        assert False, "invalid feature type"
    for wav_key in feat_dict_wavkey:
        feat_key = utils.uttlabel_to_uttkey(wav_key)
        feat_dict[feat_key] = feat_dict_wavkey[wav_key]
        if vq_dict_wavkey: vq_dict[feat_key] = vq_dict_wavkey[wav_key]
        
    # Read voice activity regions
    fa_fn = path.join("..", "data", "speechcommands.wrd")
    print("Reading:", fa_fn)
    vad_dict = utils.read_vad_from_fa(fa_fn, scale_factor=320)

    # Only keep voice active regions
    print("Extracting VAD regions:")
    feat_dict = features.extract_vad(feat_dict, vad_dict)
    if vq_dict: vq_dict = features.extract_vad(vq_dict, vad_dict)

    # Perform per speaker mean and variance normalisation
    print("Per speaker mean and variance normalisation:")
    feat_dict = features.speaker_mvn(feat_dict)

    # Write output
    print("Writing:", output_fn)
    np.savez_compressed(output_fn, **feat_dict)
    if vq_dict: np.savez_compressed(output_fn+".vq", **vq_dict)


def main(args):

    print(datetime.now())
    
    # RAW FEATURES

    # Extract Features
    feature_dir = path.join(args.save_dir, args.feature, "speechcommands")
    if not path.isdir(feature_dir):
        os.makedirs(feature_dir)
    for subset in ['dev', 'test']:
        output_fn = path.join(feature_dir, f"speechcommands.{subset}.dd.npz")
        if not path.isfile(output_fn):
            print(f"Extracting {args.feature}:")
            extract_features_for_subset(subset, args.feature, output_fn, device=args.device)
        else:
            print("Using existing file:", output_fn)


    # GROUND TRUTH WORD SEGMENTS

    # Create a ground truth word list of at least 0.5 secs and 5 characters
    fa_fn = path.join("..", "data", "speechcommands.wrd")
    list_dir = "lists"
    if not path.isdir(list_dir):
        os.makedirs(list_dir)
    list_fn = path.join(list_dir, "speechcommands.ssl.samediff.list")
    if not path.isfile(list_fn):
        utils.write_samediff_words(fa_fn, list_fn, min_frames=25, scale_factor=320)
    else:
        print("Using existing file:", list_fn)


    # Extract word segments from the Features NumPy archive
    for subset in ['dev','test']:
        input_npz_fn = path.join(feature_dir, f"speechcommands.{subset}.dd.npz")
        output_npz_fn = path.join(feature_dir, f"speechcommands.{subset}.samediff.dd.npz")
        if 'w2v' in args.feature:
            input_vq_npz_fn = input_npz_fn + ".vq.npz"
            output_vq_npz_fn = output_npz_fn + ".vq.npz"
        else:
            input_vq_npz_fn, output_vq_npz_fn = None, None
        if not path.isfile(output_npz_fn):
            print(f"Extracting {args.feature}s for same-different word tokens:", subset)
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn, 
                                    vq_npz_fn=input_vq_npz_fn,
                                    output_vq_npz_fn=output_vq_npz_fn)
        else:
            print("Using existing file:", output_npz_fn)

    print(datetime.now())
    

if __name__ == "__main__":
    
    args = utils.parse_args()
        
    main(args)
