#!/usr/bin/env python

"""
Downsample a given file using a particular technique and target dimensionality.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2018, 2019
"""

import argparse
import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal
import sys

flatten_order = "C"


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("input_npz_fn", type=str, help="input speech file")
    parser.add_argument(
        "output_npz_fn", type=str, help="output embeddings file"
        )
    parser.add_argument("n", type=int, help="number of samples")
    parser.add_argument(
        "--reduce_dims", choices=["first-k", "pca"],
        default="first-k"
        )
    parser.add_argument(
        "--technique", choices=[
            "interpolate", "resample", "rasanen", 
            "mean", "max", "stat", "vq"],
        default="resample"
        )
    parser.add_argument(
        "--frame_dims", type=int, default=None,
        help="only keep these number of dimensions"
        )
    parser.add_argument(
        "--vq_npz_fn", type=str, help="output vq file"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

#-----------------------------------------------------------------------------#
#                       Dimensional reduction Functions                       #
#    - Reduce dimensionality to generate dim-fixed features                   #
#    - Input shape : (Length, Dim)                                            #
#-----------------------------------------------------------------------------#

def get_dim_reducer(mode:str, n_dims:int, **kwargs):
    if mode == 'first-k':
        return lambda x: x[:,:n_dims]
    elif mode == 'pca':
        pca = pca_reduction(n_dims, kwargs['traindata'])
        
        def transform_pca(x):
            N, L, D = x.shape
            x = x.reshape(-1, D)
            reduce_x = pca.transform(x)
            return reduce_x.reshape(N, L, n_dims)
        
        return transform_pca
    else:
        print(f"`{mode}` is Not Implemented Reducing Method")
        return lambda x: x

def pca_reduction(n_dims:int, data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_dims)
    traindata = np.concatenate(
        [data[key].reshape(-1, data[key].shape[-1]) 
            for key in data.keys()], axis=0) # (sum(L), D)
    print(f"Start training PCA transformer for {len(traindata)} data")
    pca.fit(traindata)
    return pca
    
#-----------------------------------------------------------------------------#
#                             Sampling  Functions                             #
#    - Subsample the sequential frames. (Dim, Length)                         #
#-----------------------------------------------------------------------------#

def get_sampler(mode:str):
    if mode == 'mean':
        return meanpool
    elif mode == 'max':
        return maxpool
    elif mode == 'stat':
        return statpool
    elif mode == 'vq':
        return vqpool
    
def meanpool(y):
    return y.mean(-1)

def maxpool(y):
    return y.max(-1)

def statpool(y):
    """Pool dimensions using Statistics (Mean & Variance)
    """
    # Add gaussian ??? WHY?? @HHJ
    mean = y.mean(-1)
    std = y.std(-1)
    y = np.concatenate([mean, std], axis=0)
    return y

def probpool(y):
    raise NotImplementedError

def vqpool(y, vq):
    # Count stats
    from collections import Counter
    vq_x, vq_y = vq[:,0], vq[:,1]
    vq_x_cnt = Counter(vq_x)
    vq_y_cnt = Counter(vq_y)
    vq_x_freq = [vq_x_cnt[x] for x in vq_x]
    vq_y_freq = [vq_y_cnt[y] for y in vq_y]
    
    # Define the weights
    vq_freqs = np.array([x+y for x,y in zip(vq_x_freq, vq_y_freq)])
    probs = 1/vq_freqs
    weights = probs / probs.sum()
    
    # Pool by weights
    outs = np.dot(y, weights)
    return outs

#-----------------------------------------------------------------------------#
#                         Post processing Functions                           #
#    - Functions for pooled features to generate better representation        #
#    - Input shape : (Dims, )                                                 #
#-----------------------------------------------------------------------------#

def softdecay(y):
    u, s, v = np.linalg.svd(y)
    
    # === Soft Decay ===
    maxS = np.max(s, axis=0, keepdims=True)
    eps = 1e-7
    alpha = -0.6
    newS = -np.log(1 - alpha * (s + alpha) + eps) / alpha
    
    # === Rescaling ===
    maxNewS = np.max(newS, axis=0, keepdims=True)
    rescale_number = maxNewS / maxS
    newS = newS / rescale_number
    
    # === Transform ===
    rescale_s_dia = np.diag(newS)
    output = np.matmul(np.matmul(u, rescale_s_dia), v.T)
    
    return output

def whitening(y):
    raise NotImplementedError
    return y

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    print("Reading:", args.input_npz_fn)
    input_npz = np.load(args.input_npz_fn)
    d_frame = input_npz[sorted(input_npz.keys())[0]].shape[1]

    print("Frame dimensionality:", d_frame)
    if args.frame_dims is not None and args.frame_dims < d_frame:
        d_frame = args.frame_dims
        print("Reducing frame dimensionality:", d_frame)
    reducer = get_dim_reducer(
        args.reduce_dims, 
        args.frame_dims,
        traindata = input_npz)

    print("Downsampling:", args.technique)
    output_npz = {}
    vq_npz = None
    for key in input_npz:

        # Limit input dimensionailty
        y = reducer(input_npz[key]).T
        # y = input_npz[key][:, :args.frame_dims].T

        # Downsample
        if args.technique == "interpolate":
            x = np.arange(y.shape[1])
            f = interpolate.interp1d(x, y, kind="linear")
            x_new = np.linspace(0, y.shape[1] - 1, args.n)
            y_new = f(x_new).flatten(flatten_order) #.flatten("F")
        elif args.technique == "resample":
            y_new = signal.resample(
                y, args.n, axis=1
                ).flatten(flatten_order) #.flatten("F")
        elif args.technique == "rasanen":
            # Taken from Rasenen et al., Interspeech, 2015
            n_frames_in_multiple = int(np.floor(y.shape[1] / args.n)) * args.n
            y_new = np.mean(
                y[:, :n_frames_in_multiple].reshape((d_frame, args.n, -1)),
                axis=-1
                ).flatten(flatten_order) #.flatten("F")
        elif "vq" in args.technique:
            if vq_npz is None:
                vq_npz = np.load(args.vq_npz_fn)
            vq = vq_npz[key]
            sampler = get_sampler(args.technique)
            y_reduced = sampler(y, vq)
            y_new = y_reduced.flatten(flatten_order)
        elif args.n == 1:
            sampler = get_sampler(args.technique)
            y_reduced = sampler(y)
            y_new = y_reduced.flatten(flatten_order)
        else:
            raise NotImplementedError

        # This was done in Rasenen et al., 2015, but didn't help here
        # last_term = args.n/3. * np.log10(y.shape[1] * 10e-3)
        # Not sure if the above should be in frames or ms
        # y_new = np.hstack([y_new, last_term])
        
        # Save result
        output_npz[key] = y_new

    print(
        "Output dimensionality:",
        output_npz[sorted(output_npz.keys())[0]].shape[0]
        )

    print("Writing:", args.output_npz_fn)
    np.savez_compressed(args.output_npz_fn, **output_npz)


if __name__ == "__main__":
    main()
