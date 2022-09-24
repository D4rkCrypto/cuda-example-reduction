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
| reduce1  | 10000  | 62.777    | 62.464      | 0.001234  |
| reduce2  | 10000  | 47.744    | 47.904      | 0.000621  |
| reduce3  | 10000  | 46.681    | 46.848      | 0.000522  |
| reduce4  | 10000  | 26.134    | 26.240      | 0.001109  |
| reduce5  | 10000  | 21.237    | 21.408      | 0.000857  |
| reduce6  | 10000  | 21.221    | 21.376      | 0.001036  |
| reduce7  | 10000  | 21.275    | 21.344      | 0.001480  |
