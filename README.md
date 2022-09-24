# CUDA Example of Reduction

## Introduction

CUDA example of parallel reduction, see [NVIDIA slide](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf).

## Prerequisites

- CLion (recommended) or other IDEs
- CUDA
- Boost headers, see details in [Boost website](https://www.boost.org/)

## Build

- CMake, no additional arguments required

## Test

- run reduction executable

## Performance Evaluation

- NVIDIA GeForce RTX® 3090 Ti

| function | trials | mean (us) | median (us) | std. dev. |
|----------|--------|-----------|-------------|-----------|
| reduce1  | 10000  | 62.692    | 62.336      | 0.001405  |
| reduce2  | 10000  | 47.133    | 47.104      | 0.000527  |
| reduce3  | 10000  | 46.088    | 46.080      | 0.000584  |
| reduce4  | 10000  | 25.832    | 25.600      | 0.000732  |
| reduce5  | 10000  | 21.233    | 21.408      | 0.000713  |
| reduce6  | 10000  | 21.208    | 21.376      | 0.000570  |
| reduce7  | 10000  | 21.181    | 21.344      | 0.000560  |
