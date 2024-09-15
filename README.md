# ShaneGPUCountMinSketch

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kchu25.github.io/ShaneGPUCountMinSketch.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/ShaneGPUCountMinSketch.jl/dev/)
[![Build Status](https://github.com/kchu25/ShaneGPUCountMinSketch.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/ShaneGPUCountMinSketch.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/ShaneGPUCountMinSketch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/ShaneGPUCountMinSketch.jl)


The primary use of this package is to count motif instances
as recorded from the sparse code. This package uses GPU (CUDA) to speed up the enumerations on a probabilistic data structure called Count-Min-Sketch.

We will define a *configuration* $c_K$ as a $2K+1$ $(K = 1,2,3,...)$ tuple. For a  where:
* the component $c[2k+1]$, where $1\leq k \leq K$, documents the filter index.
* the component $c[2k]$, where $1\leq k \leq K-1$, documents the distance (e.g. number of nucleotide in between) the filters $c[2(k-1)+1]$ and $c[2k+1]$.

The subroutine provided by this package is:

```
obtain_enriched_configurations(
        nz_dict::Dict{Int, Vector{CartesianIndex{2}}},
        num_fils::Int, 
        fil_len::Int;
        min_count=default_min_count
)
```

where `nz_dict` is a dict where 
* its keys are Integers (type `Int`) that records the *sequence index*
* *num_fils*: the number of filters that the user want to recorded in a *configuration* (i.e. the variable $K$ in the definition of configuration).
* *fil_len*: the filter length
* *min_count*: the minimum number of counts for a configuration to be considered valid.
