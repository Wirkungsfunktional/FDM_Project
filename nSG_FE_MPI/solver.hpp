#ifndef SOLVER_H_
#define SOLVER_H_





template<FUNC>
class solver {

private:
    int world_rank;
    int world_size;
    int reduced_sys_size;
    double *u;
    double ta, te, dt;
    double lbound, rbound;

public:
    explicit solver(int N, int size, int rank, double t1, double t2, double k) {
        world_rank = rank;
        world_size = size;
        reduced_sys_size = N / world_size;
        u = new double[reduced_sys_size];
        ta = t1;
        te = t2;
        dt = k;
    }
    ~solver() {
        delete[] u;
    }
    void run();
};


template<FUNC>
void solver<FUNC>::run() {
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






}













#endif //SOLVER_H_
