#!/usr/bin/env python

"""
Extract Self-supervised features for the Buckeye dataset.

Author: Jeongkyun Park
Contact: park32323@gmail.com
Date: 2023
"""

from datetime import datetime
from os import path
from tqdm import tqdm
import argparse
import numpy as np
import os
import glob
import sys

sys.path.append("..")

from paths import buckeye_datadir
import features
import utils


def extract_features_for_subset(subset, feat_type, output_fn, device='cpu'):
    """
    Extract specified features for a subset.

    The `feat_type` parameter can be "w2v" or "hb".
    """

    # Speakers for subset
    speaker_fn = path.join(
        "..", "data", "buckeye_" + subset + "_speakers.list"
        )
    print("Reading:", speaker_fn)
    speakers = set()
    with open(speaker_fn) as f:
        for line in f:
            speakers.add(line.strip())
    print("Speakers:", ", ".join(sorted(speakers)))

    # Define key
    def keygen(fn):
        wav_nm = fn.split('/')[-1][:-4]
        utt_key = f"{wav_nm[:3]}_{wav_nm[-3:]}"
        return utt_key

    # Raw feature
    feat_dict = {}
    vq_dict = {}
    paths = []
    for spk in speakers:
        datadir = path.join(buckeye_datadir, spk)
        paths.extend(sorted(glob.glob(path.join(datadir, '*.wav'))))
    postprocess = utils.get_postprocessor(feat_type)
    if "w2v" in feat_type:
        feat_dict_wavkey, vq_dict_wavkey = features.extract_w2v_dir(
            paths=paths,
            vad_dict=vad_dict,
            postprocess=postprocess,
            key_generator=keygen
            )
        vq_dict.update(vq_dict_wavkey)
    elif "hb" in feat_type:
        feat_dict_wavkey = features.extract_w2v_dir(
            paths=paths,
            vad_dict=vad_dict,
            postprocess=postprocess,
            key_generator=keygen
            )
    else:
        assert False, "invalid feature type"
    feat_dict.update(feat_dict_wavkey)

    # Read voice activity regions
    fa_fn = path.join("..", "data", "buckeye_english.wrd")
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

    # Extract Features for the different sets
    feature_dir = path.join(args.save_dir, args.feature, "buckeye")
    if not path.isdir(feature_dir):
        os.makedirs(feature_dir)
    for subset in ["zs"]: # ["devpart1", "devpart2", "zs"]:
        output_fn = path.join(feature_dir, subset + ".dd.npz")
        if not path.isfile(output_fn):
            print(f"Extracting {args.feature}:", subset)
            extract_features_for_subset(subset, args.feature, output_fn, device=args.device)
        else:
            print("Using existing file:", output_fn)


    # GROUND TRUTH WORD SEGMENTS

    # Create a ground truth word list of at least 50 frames and 5 characters
    fa_fn = path.join("..", "data", "buckeye_english.wrd")
    list_dir = "lists"
    if not path.isdir(list_dir):
        os.makedirs(list_dir)
    list_fn = path.join(list_dir, "buckeye.samediff.list")
    if not path.isfile(list_fn):
        utils.write_samediff_words(fa_fn, list_fn, min_frames=25, scale_factor=320)
    else:
        print("Using existing file:", list_fn)

    # Extract word segments from the Feature NumPy archives
    for subset in ["zs"]: #["devpart1", "devpart2", "zs"]:
        input_npz_fn = path.join(feature_dir, subset + ".dd.npz")
        output_npz_fn = path.join(feature_dir, subset + ".samediff.dd.npz")
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
