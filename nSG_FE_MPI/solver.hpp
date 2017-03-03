#ifndef SOLVER_H_
#define SOLVER_H_

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <complex>



struct Heat_eq
{
    static inline int topo_left_bound(int rank, int size){
        return (size + rank - 1) % size;
    }
    static inline int topo_right_bound(int rank, int size){
        return (rank + 1) % size;
    }
    static inline double set_initial_values(double x) {
        return std::sin(M_PI * x);
    }
    static inline double eval_with_left_bound(double* u, double b, double r, double k) {
        return u[0] + r*(b - 2*u[0] + u[1]);
    }
    static inline double eval_with_right_bound(double* u, double b, double r, int size, double k) {
        return u[size-1] + r*(u[size-2] - 2*u[size-1] + b);
    }
    static inline double eval(double* u, int i, double r, double k) {
        return u[i] + r*(u[i-1] - 2*u[i] + u[i+1]);
    }
    static inline void write_data_to_file(double* u, int size){
        for (int i=0; i<size; i++) {
            std::cout << u[i] << " ";
        }
        std::cout << "\n";
    }
};

struct nSG
{
    static inline int topo_left_bound(int rank, int size){
        return (size + rank - 1) % size;
    }
    static inline int topo_right_bound(int rank, int size){
        return (rank + 1) % size;
    }
    static inline std::complex<double> set_initial_values(double x) {
        std::complex<double> I(0.0,1.0);
        return std::exp(I * x);
    }
    static inline std::complex<double> eval_with_left_bound(std::complex<double>* u, std::complex<double> b, double r, double k) {
        std::complex<double> I(0,1);
        return u[0] + I*r*(b - 2.0*u[0] + u[1]) - I*0.5*k*std::abs(u[0])*std::abs(u[0])*u[0];
    }
    static inline std::complex<double> eval_with_right_bound(std::complex<double>* u, std::complex<double> b, double r, int size, double k) {
        std::complex<double> I(0,1);
        return u[size-1] + I*r*(u[size-2] - 2.0*u[size-1] + b) - I*0.5*k*std::abs(u[size-1])*std::abs(u[size-1])*u[size-1];
    }
    static inline std::complex<double> eval(std::complex<double>* u, int i, double r, double k) {
        std::complex<double> I(0,1);
        return u[i] + I*r*(u[i-1] - 2.0*u[i] + u[i+1]) - I*0.5*k*std::abs(u[i])*std::abs(u[i])*u[i];
    }
    static inline void write_data_to_file(std::complex<double>* u, int size){
        for (int i=0; i<size; i++) {
            std::cout << std::abs(u[i]) << " ";
        }
        std::cout << "\n";
    }
};






template<class FUNC, typename DATA_T>
class solver {

private:
    int world_rank;
    int world_size;
    int reduced_sys_size;
    DATA_T *u;
    double ta, te, dt;
    double h, r;
    double xa, xe;
    DATA_T lbound, rbound;
    int left_nb, right_nb;

public:
    explicit solver(int N, int size, int rank, double t1, double t2, double x1, double x2) {
        std::cout << "Init " << rank << "\n";
        world_rank = rank;
        world_size = size;
        reduced_sys_size = N / world_size;
        u = new DATA_T[reduced_sys_size];
        ta = t1;
        te = t2;
        xa = x1;
        xe = x2;
        h = (xe-xa) / ((double) (N+1)) ;
        dt = h*h/2/10000;
        r = dt/h/h;
        std::cout << dt << "\n";
        left_nb = FUNC::topo_left_bound(world_rank, world_size);
        right_nb = FUNC::topo_right_bound(world_rank, world_size);
    }
    ~solver() {
        delete[] u;
    }
    void run();
};


template<class FUNC, typename DATA_T>
void solver<FUNC, DATA_T>::run() {

        #pragma omp parallel for
        for (int k=0; k<reduced_sys_size; k++) {
            u[k] = FUNC::set_initial_values( (world_rank*reduced_sys_size + k)*h + xa );
        }
        std::cout << "Data initiated in " << world_rank << "\n";

        while (ta < te) {
            MPI_Send(&u[0], 1, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD);
            MPI_Send(&u[reduced_sys_size-1], 1, MPI_DOUBLE, right_nb, 0, MPI_COMM_WORLD);
            MPI_Recv(&rbound, 1, MPI_DOUBLE, right_nb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&lbound, 1, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            u[0] = FUNC::eval_with_left_bound(u, lbound, r, dt);
            u[reduced_sys_size-1] = FUNC::eval_with_right_bound(u, rbound, r, reduced_sys_size, dt);
            for (int i=1; i<reduced_sys_size-1; i++) {
                u[i] = FUNC::eval(u, i, r, dt);
            }
            ta += dt;
        }
        std::cout << "complete eval in " << world_rank << "\n";


        if (world_rank == 0) {
            FUNC::write_data_to_file(u, reduced_sys_size);
            for (int i=1; i< world_size; i++) {
                MPI_Recv(u, reduced_sys_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                FUNC::write_data_to_file(u, reduced_sys_size);
            }
        } else {
            MPI_Send(u, reduced_sys_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }


}


#endif //SOLVER_H_
