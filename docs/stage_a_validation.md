# Stage A Validation Note

## Purpose

Stage A uses UTD-MHAD as a computational validation dataset, not as a baseball
performance dataset.

The role of Stage A is to answer:
- can raw joint captures be loaded and standardized reliably
- can they be converted into stable trajectory matrices and feature matrices
- can derivatives be estimated consistently
- can PySINDy produce sparse coefficient matrices and equation structures
- how sensitive are those coefficients to preprocessing and feature choices

It does not answer:
- whether UTD-MHAD motions are faithful stand-ins for baseball pitching
- whether UTD-MHAD swings replicate athlete-quality batting mechanics
- whether discovered equations should be interpreted as baseball biomechanics

## What Counts As A Successful Stage A Output

The following are valid Stage A outputs:
- standardized joint trajectories
- processed clip manifests
- state matrices `X(t)`
- derivative matrices `dX/dt`
- sparse coefficient matrices
- equation files
- configuration-sensitive comparisons across preprocessing and model settings

These outputs validate the pipeline infrastructure.

## What Not To Conclude

Do not treat Stage A results as:
- evidence of elite or realistic baseball mechanics
- evidence that a discovered coefficient is baseball-specific
- evidence that poor reconstruction on UTD-MHAD invalidates the overall approach

UTD-MHAD contains non-athlete subjects performing generic actions. That makes it
useful for capture and matrix-generation testing, but limited for sport-specific
interpretation.

## Current Interpretation Standard

When reading Stage A artifacts in `results/equations/`, `results/evaluation/`,
and `results/plots/`, use the following interpretation standard:

- `results/equations/`: sparse structure and coefficient recovery worked or failed
- `results/evaluation/`: model stability and configuration sensitivity
- `results/plots/`: short-horizon sanity checks, not biomechanics evidence

The correct Stage A question is:

"Can this pipeline turn captured joints into reproducible sparse model artifacts?"

The baseball question belongs to Stage B, using baseball-specific data.
