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
import sys

sys.path.append("..")

from paths import buckeye_datadir
import features
import utils


def extract_features_for_subset(subset, feat_type, output_fn):
    """
    Extract specified features for a subset.

    The `feat_type` parameter can be "mfcc" or "fbank".
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

    # Raw features
    feat_dict = {}
    print("Extracting features per speaker:")
    for speaker in sorted(speakers):
        
        if "w2v" in feat_type or "hb" in feat_type:
            continue
        # if "w2v" in feat_type:
        #     postprocess = utils.get_postprocessor(feat_type)
        #     speaker_feat_dict, speaker_vq_dict = features.extract_w2v_dir(
        #         path.join(buckeye_datadir, speaker),
        #         postprocess=postprocess,
        #         )
        # elif "hb" in feat_type:
        #     postprocess = utils.get_postprocessor(feat_type)
        #     speaker_feat_dict = features.extract_hb_dir(
        #         path.join(buckeye_datadir, speaker),
        #         postprocess=postprocess,
        #         )
        else:
            assert False, "invalid feature type"
        for wav_key in speaker_feat_dict:
            feat_dict[speaker + "_" + wav_key[3:]] = speaker_feat_dict[wav_key]

    # Read voice activity regions
    # NOTE(JK) Read pre-defined word boundary and return it. (Convert second to frame, assuming 1/100 scale)
    fa_fn = path.join("..", "data", "buckeye_english.wrd")
    print("Reading:", fa_fn)
    vad_dict = utils.read_vad_from_fa(
        fa_fn, 
        frame_indices=False
    )

    if "w2v" in feat_type:
        # NOTE(JK) We do extract the feature in here. 
        # We segment the waveform first, forwarding the total wav costs too much.
        vq_dict = {}
        postprocess = utils.get_postprocessor(feat_type)
        for speaker in sorted(speakers):
            feat_dict_, vq_dict_ = features.extract_w2v_dir(
                path.join(buckeye_datadir, speaker),
                vad_dict=vad_dict,
                postprocess=postprocess,
                )
            feat_dict.update(feat_dict_)
            vq_dict.update(vq_dict_)
    elif "hb" in feat_type:
        vq_dict = {}
        postprocess = utils.get_postprocessor(feat_type)
        for speaker in sorted(speakers):
            feat_dict_ = features.extract_hb_dir(
                path.join(buckeye_datadir, speaker),
                vad_dict=vad_dict,
                postprocess=postprocess,
                )
            feat_dict.update(feat_dict_)
            # vq_dict.update(vq_dict_)
    
    # Perform per speaker mean and variance normalisation
    print("Per speaker mean and variance normalisation:")
    feat_dict = features.speaker_mvn(feat_dict)

    # Write output
    print("Writing:", output_fn)
    np.savez_compressed(output_fn, **feat_dict)
    if vq_dict:
        np.savez_compressed(output_fn+".vq", **vq_dict)


def main(feature:str, savedir:str=None):

    assert 'w2v' in feature or 'hb' in feature
    index_factor = 0.5

    print(datetime.now())

    # RAW FEATURES

    # Extract Features for the different sets
    if savedir:
        feature_dir = path.join(savedir, feature, "buckeye")
    else:
        feature_dir = path.join(feature, "buckeye")
    for subset in ["zs"]: # ["devpart1", "devpart2", "zs"]:
        if not path.isdir(feature_dir):
            os.makedirs(feature_dir)
        output_fn = path.join(feature_dir, subset + ".dd.npz")
        if not path.isfile(output_fn):
            print(f"Extracting {feature}:", subset)
            extract_features_for_subset(subset, feature, output_fn)
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
        utils.write_samediff_words(fa_fn, list_fn)
    else:
        print("Using existing file:", list_fn)

    # NOTE(JK) Just limit some feature segments and save them for samediff evaluation.

    # Extract word segments from the Feature NumPy archives
    for subset in ["zs"]: #["devpart1", "devpart2", "zs"]:
        input_npz_fn = path.join(feature_dir, subset + ".dd.npz")
        output_npz_fn = path.join(feature_dir, subset + ".samediff.dd.npz")
        if 'w2v' in feature:
            input_vq_npz_fn = input_npz_fn + ".vq.npz"
            output_vq_npz_fn = output_npz_fn + ".vq.npz"
        else:
            input_vq_npz_fn, output_vq_npz_fn = None, None
        if not path.isfile(output_npz_fn):
            print(f"Extracting {feature}s for same-different word tokens:", subset)
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn, 
                                    vq_npz_fn=input_vq_npz_fn,
                                    output_vq_npz_fn=output_vq_npz_fn,
                                    index_factor=index_factor)
        else:
            print("Using existing file:", output_npz_fn)

    print(datetime.now())


if __name__ == "__main__":
    
    import sys
    
    savedir=None
    try:
        args = sys.argv[1:]
        if len(args) == 1:
            feature = args[0]
        elif len(args) == 2:
            feature, savedir = args
    except IndexError:
        print("Pass the feature type. e.g. `./extract_features_buckeye w2v_1 ./data`")

    main(feature, savedir)
