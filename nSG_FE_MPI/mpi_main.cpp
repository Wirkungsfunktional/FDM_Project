
#include <mpi.h>
#include "solver.hpp"




int main(int argc, char* argv[]) {

    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int N = 1000;
    solver<nSG, std::complex<double> > s(N,world_size,world_rank,0.0,0.001, -M_PI, M_PI);
    s.run();




    MPI_Finalize();

    return 0;
}
