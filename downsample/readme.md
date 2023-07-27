Downsampled Acoustic Word Embeddings
====================================


Overview
--------
MFCCs are downsampled to obtain acoustic word embeddings. These are evaluated
using same-different evaluation.


Downsampling
------------
Perform downsampling on MFCCs without deltas:

    # Devpart2
    n_samples=10
    mkdir -p exp/devpart2
    ./downsample.py --technique resample --frame_dims 13 \
        ../features/mfcc/buckeye/devpart2.samediff.dd.npz \
        exp/devpart2/samediff.mfcc.downsample_${n_samples}.npz \
        ${n_samples}

    # ZeroSpeech
    n_samples=10
    mkdir -p exp/zs
    ./downsample.py --technique resample --frame_dims 13 \
        ../features/mfcc/buckeye/zs.samediff.dd.npz \
        exp/zs/samediff.mfcc.downsample_${n_samples}.npz \
        ${n_samples}

    # Xitsonga
    n_samples=10
    mkdir -p exp/xitsonga
    ./downsample.py --reduce_dims first-k --technique resample --frame_dims 13 \
        ../features/mfcc/xitsonga/xitsonga.samediff.dd.npz \
        exp/xitsonga/samediff.mfcc.downsample_${n_samples}.npz \
        ${n_samples}

    # To activate VQ files
    n_samples=10
    mkdir -p exp/xitsonga
    ./downsample.py --reduce_dims first-k --technique resample --frame_dims 13 \
        --vq_npz_fn ../features/w2v_11/xitsonga/xitsonga.samediff.dd.npz.vq.npz \
        ../features/w2v_11/xitsonga/xitsonga.samediff.dd.npz \
        exp/xitsonga/samediff.w2v_11.downsample_${n_samples}.npz \
        ${n_samples}


Evaluation
----------
Evaluate and analyse downsampled MFCCs without deltas:

    # Devpart2
    n_samples=10
    ./eval_samediff.py --mvn \
        exp/devpart2/samediff.mfcc.downsample_${n_samples}.npz
    ./analyse_embeds.py --normalize --word_type \
        because,yknow,people,something,anything,education,situation \
        exp/devpart2/samediff.mfcc.downsample_${n_samples}.npz

    # ZeroSpeech
    n_samples=10
    ./eval_samediff.py --mvn \
        exp/zs/samediff.mfcc.downsample_${n_samples}.npz
    ./analyse_embeds.py --normalize --word_type \
        because,yknow,people,something,anything,education,situation \
        exp/zs/samediff.mfcc.downsample_${n_samples}.npz

    # Xitsonga
    n_samples=10
    ./eval_samediff.py --mvn \
        exp/xitsonga/samediff.mfcc.downsample_${n_samples}.npz
    ./analyse_embeds.py --normalize --word_type \
        kombisa,swilaveko,kahle,swinene,xiyimo,fanele,naswona,xikombelo \
        exp/xitsonga/samediff.mfcc.downsample_${n_samples}.npz


Results
-------
Devpart2 downsampled MFCCs without deltas (dimensionality=130):
    
    Average precision: 0.2434209747796554
    Precision-recall breakeven: 0.2854930304594734

ZeroSpeech downsampled MFCCs without deltas + mvn (dimensionality=130):

    Average precision: 0.21282214281152606
    Precision-recall breakeven: 0.2731139747278869

Xitsonga downsampled MFCCs without deltas (dimensionality=130):

    Average precision: 0.1132638941316873
    Precision-recall breakeven: 0.1845537016037583
