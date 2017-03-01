#ifndef SOLVER_H_
#define SOLVER_H_



struct PDE
{
    static inline int topo_left_bound(int rank, int size);
    static inline int topo_right_bound(int rank, int size);
    static inline double set_initial_values(double x);
    static inline double eval_with_left_bound(double* u, double b);
    static inline double eval_with_left_bound(double* u, double b);
    static inline double eval(double* u):
    static inline void write_data_to_file(double* u);
};






template<FUNC>
class solver {

private:
    int world_rank;
    int world_size;
    int reduced_sys_size;
    double *u;
    double ta, te, dt;
    double lbound, rbound;
    int left_nb, right_nb;

public:
    explicit solver(int N, int size, int rank, double t1, double t2, double k) {
        world_rank = rank;
        world_size = size;
        reduced_sys_size = N / world_size;
        u = new double[reduced_sys_size];
        ta = t1;
        te = t2;
        dt = k;
        left_nb = FUNC::topo_left_bound(world_rank, world_size);
        right_nb = FUNC::topo_right_bound(world_rank, world_size);
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
            #pragma omp parallel for
            for (int k=0; k<reduced_sys_size; k++) {
                u[k] = FUNC::set_initial_values( i*reduced_sys_size + k);
            }
            MPI_Send(u, reduced_sys_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            std::cout << "Send data from " << world_rank << " to " << i << "\n";
        }

        while (ta < te) {
            MPI_Send(&u[0], 1, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD);
            MPI_Send(&u[reduced_sys_size-1], 1, MPI_DOUBLE, right_nb, 0, MPI_COMM_WORLD);
            MPI_Recv(&rbound, 1, MPI_DOUBLE, right_nb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&lbound, 1, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            u[0] = FUNC::eval_with_left_bound(u, lbound);
            u[reduced_sys_size-1] = FUNC::eval_with_right_bound(u, rbound);
            for (int i=1; i<reduced_sys_size-1; i++) {
                u[i] = FUNC::eval(u, i);
            }
            ta += dt;
        }

        for (int i=1; i< world_size; i++) {
            MPI_Recv(u, reduced_sys_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            FUNC::write_data_to_file(u);
        }

    } else {
            MPI_Recv(&u, reduced_sys_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << world_rank << " Recieve data from  0\n";

            while (ta < te) {
                MPI_Send(&u[0], 1, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD);
                MPI_Send(&u[reduced_sys_size-1], 1, MPI_DOUBLE, right_nb, 0, MPI_COMM_WORLD);
                MPI_Recv(&rbound, 1, MPI_DOUBLE, right_nb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&lbound, 1, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                u[0] = FUNC::eval_with_left_bound(u, lbound);
                u[reduced_sys_size-1] = FUNC::eval_with_right_bound(u, rbound);
                #pragma omp parallel for
                for (int i=1; i<reduced_sys_size-1; i++) {
                    u[i] = FUNC::eval(u, i);
                }
                ta += dt;
            }
            MPI_Send(u, reduced_sys_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    }






}













#endif //SOLVER_H_
