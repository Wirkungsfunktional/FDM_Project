#!/bin/bash

shopt -s expand_aliases
source ~/.bashrc

for i in 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001; do
  /home/mk/Programming/petsc-3.7.5/arch-linux2-cxx-debug/bin/mpiexec -n 4 ./main 4096 $i 1 1;
done
