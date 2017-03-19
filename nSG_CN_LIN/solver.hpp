#ifndef SOLVER_H_
#define SOLVER_H_

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <complex>
#include <fstream>
#include "../misc/io_utility.hpp"


template<class FUNC, typename DATA_T>
class solver {

private:
    int world_rank;
    int world_size;
    int reduced_sys_size, real_size;
    DATA_T *u, *u0;
    FUNC *pde;
    double ta, te, dt;
    DATA_T lbound, rbound;
    int left_nb, right_nb;

public:
    explicit solver(int N, int size, int rank, double t1, double t2, double x1, double x2) {
        std::cout << "Init " << rank << "\n";
        world_rank = rank;
        world_size = size;
        left_nb = FUNC::topo_left_bound(world_rank, world_size);
        right_nb = FUNC::topo_right_bound(world_rank, world_size);
        pde->set_world_spec(world_size, world_rank);


        real_size = pde->get_N();
        reduced_sys_size = real_size / world_size;
        u = new DATA_T[reduced_sys_size];
        u0 = new DATA_T[reduced_sys_size];

        ta = pde->get_ta();
        te = pde->get_te();
        dt = pde->get_dt();
    }
    ~solver() {
        delete[] u;
        delete[] u0;
    }
    void run();
};


template<class FUNC, typename DATA_T>
void solver<FUNC, DATA_T>::run() {


        pde->set_initial_values(u);
        std::cout << "Data initiated in " << world_rank << "\n";

        while (ta < te) {
            MPI_Send(&u[0], 1, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD);
            MPI_Send(&u[reduced_sys_size-1], 1, MPI_DOUBLE, right_nb, 0, MPI_COMM_WORLD);
            MPI_Recv(&rbound, 1, MPI_DOUBLE, right_nb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&lbound, 1, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pde->eval(u, u0, lbound, rbound);
            ta += dt;
        }
        std::cout << "complete eval in " << world_rank << "\n";


        if (world_rank == 0) {

            pde->write_data_to_file(u, 0);
            for (int i=1; i< world_size; i++) {
                MPI_Recv(u, reduced_sys_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pde->write_data_to_file(u, i);
            }
        } else {
            MPI_Send(u, reduced_sys_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }


}


#endif //SOLVER_H_
