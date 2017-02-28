#include <iostream>
#include <mpi.h>




#define N 100000


double fill_u0(int i) {
    return (double) i;
}

int left_neigbour(int n, int size) {
    return (size + n - 1) % size;
}
int right_neigbour(int n, int size) {
    return (n+1) % size;
}
double left_bound(double* v, double b) {
    return b;
}
double right_bound(double* v, double b) {
    return b;
}
double update(double* v, int i) {
    return v[i];
}


int main(int argc, char* argv[])
{
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int reduced_sys_size = N / world_size;
    double u[reduced_sys_size];
    double ta = 0;
    double te = 1;
    double dt = 0.00001;
    double lbound;
    double rbound;

    if (world_rank == 0) {

        for (int i=1; i<world_size; i++) {
            for (int k=0; k<reduced_sys_size; k++) {
                u[k] = fill_u0( i*reduced_sys_size + k);
            }
            MPI_Send(u, reduced_sys_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            std::cout << "Send data from " << world_rank << " to " << i << "\n";
        }

        while (ta < te) {
            MPI_Send(&u[0], 1, MPI_DOUBLE, left_neigbour(world_rank, world_size), 0, MPI_COMM_WORLD);
            MPI_Send(&u[reduced_sys_size-1], 1, MPI_DOUBLE, right_neigbour(world_rank, world_size), 0, MPI_COMM_WORLD);
            MPI_Recv(&rbound, 1, MPI_DOUBLE, right_neigbour(world_rank, world_size), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&lbound, 1, MPI_DOUBLE, left_neigbour(world_rank, world_size), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            u[0] = left_bound(u, lbound);
            u[reduced_sys_size-1] = right_bound(u, rbound);
            for (int i=1; i<reduced_sys_size-1; i++) {
                u[i] = update(u, i);
            }
            ta += dt;
        }

        for (int i=1; i< world_size; i++) {
            MPI_Recv(u, reduced_sys_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Recieved Result from " << i << "\n";
        }

    } else {
            MPI_Recv(&u, reduced_sys_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << world_rank << " Recieve data from  0\n";




            while (ta < te) {
                MPI_Send(&u[0], 1, MPI_DOUBLE, left_neigbour(world_rank, world_size), 0, MPI_COMM_WORLD);
                MPI_Send(&u[reduced_sys_size-1], 1, MPI_DOUBLE, right_neigbour(world_rank, world_size), 0, MPI_COMM_WORLD);
                MPI_Recv(&rbound, 1, MPI_DOUBLE, right_neigbour(world_rank, world_size), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&lbound, 1, MPI_DOUBLE, left_neigbour(world_rank, world_size), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                u[0] = left_bound(u, lbound);
                u[reduced_sys_size-1] = right_bound(u, rbound);
                for (int i=1; i<reduced_sys_size-1; i++) {
                    u[i] = update(u, i);
                }
                ta += dt;
            }
            MPI_Send(u, reduced_sys_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    }





    MPI_Finalize();
    return 0;
}
