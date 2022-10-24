#!/bin/bash

nvcc -O3 -use_fast_math -o galaxies framework.cu && ./galaxies
