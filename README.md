pipeline:

## 1. Goal: segment tooth from cbct

method: 
    MedSAM2 (github)


## 2. Goal: MAE modeling of single Tooth

observation:
    a tooth is a simple surface when viewd from center-of-mass
    HEALPix: see attached pdf (Górski 2005)

method:
    HEALPix NESTED ordering can translate an stl surface to a 1-d array 
        -> only have to do 1d masked autoencoding


## 3. Goal: FEA analysis

observation:
    if a corrupted tooth have a sufficiently large portion intact,
    our MAE can render a full tooth

    subtracting from rentered tooth the corrupted GT gives us a Prosthodontics treatment

method:
    MFEM (github)


the advantage of this pipeline is its economy on the scarce bad-tooth data, and leverage abundant full-teeth datasets and models. 

a pre-experiment using only existing stl- format single tooth annotation yields a working tooth MAE
and such data can be obtained via [full mouth CBCT + tooth segmentation], the latter was actively researched


this codebase is only a pre-experiment validating the 2nd step. 