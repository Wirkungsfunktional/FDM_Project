#!/bin/bash

shopt -s expand_aliases
source ~/.bashrc

for i in 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768; do
  /home/mk/Programming/petsc-3.7.5/arch-linux2-cxx-debug/bin/mpiexec -n 4 ./main $i 0.0005 0.1 1;
done
