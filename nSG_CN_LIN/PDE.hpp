#ifndef PDE_H_
#define PDE_H_



class nSG_LIN {

private:
    double l = 0.5;
    int N, reduced_sys_size;
    double ta, te;
    double xa, xe;
    double h, dt, r;
    int world_size, world_rank;
    std::complex<double> I;


public:
    explicit nSG_LIN() {
        l = 0.5;
        N = 1000;
        ta = 0.0;
        te = 0.001;
        xa = -M_PI;
        xe = M_PI;
        h = (xe-xa) / (double (N+1)) ;
        dt = 0.01;
        r = dt/(h*h);
        I = std::complex<double>(0.0, 1.0);
    }
    inline void set_world_spec(int size, int rank);
    static inline int topo_left_bound(int rank, int size);
    static inline int topo_right_bound(int rank, int size);

    inline double get_ta();
    inline double get_te();
    inline double get_dt();
    inline double get_N();

    inline void set_initial_values(std::complex<double> *u);
    inline std::complex<double> eval(std::complex<double>* u, std::complex<double>* u0,
                std::complex<double> lb, std::complex<double> rb);
    inline void write_data_to_file(std::complex<double>* u);

};



static inline int nSG_LIN::topo_left_bound(int rank, int size){
    return (size + rank - 1) % size;
}
static inline int nSG_LIN::topo_right_bound(int rank, int size){
    return (rank + 1) % size;
}




inline void nSG_LIN::set_initial_values(std::complex<double> *u) {
    for (int i=0; i<reduced_sys_size; i++) {
        u[i] = std::exp(I * ((world_rank*reduced_sys_size + (i+1))*h + xa));
    }
}



inline std::complex<double> nSG_LIN::eval(std::complex<double>* u, std::complex<double>* un,
            std::complex<double> lb, std::complex<double> rb) {

    double a;
    for (int i=0;i<reduced_sys_size;i++) {
        un[i] = u[i];
    }

    for (int i=0; i<reduced_sys_size; i++) {
        a = 1 - r - l*dt/2 * std::abs(u[i])*std::abs(u[i]);

        un[0] = (f(u, 0) - 2*r*(lb + u0[1]))/a;
        for (k=1; k<size-1;k++) {
            u0[k] = (f(u, 0) - 2*r*(u0[k-1] + u0[k+1]))/a;
        }
        u0[k] = (f(u, 0) - 2*r*(u0[size-2] + rb))/a(u);
    }


    return u0;
}


inline void write_data_to_file(std::complex<double>* u){
    std::ofstream myfile;
    std::complex<double> u0;
    double x;
    myfile.open("../data/Results.txt");     //TODO:Change name by input
    // Header
    myfile << "5" << "\n";
    myfile << "nSG with periodic bound(-PI,PI),FE_Solver" << "\n";
    myfile << "Date: " << get_time_and_data() << "\n";
    myfile << "N=" << N << ";te=" << te << ";dt=" << dt << "\n";
    myfile << "u0.re,u0.im,u(x).re,u(x).im,x" << "\n";
    // Data
    for (int i=0; i<N; i++) {
        x = (i+1)*h - M_PI;
        u0 = set_initial_values(x);
        myfile  << u0.real() << ";" << u0.imag() << ";"
                << u[i].real() << ";" << u[i].imag() << ";"
                << x << "\n";
    }
    myfile.close();
}






inline void nSG_LIN::set_world_spec(int size, int rank) {
    world_size = size;
    world_rank = rank;
}

inline double nSG_LIN::get_ta() {
    return ta;
}
inline double nSG_LIN::get_te() {
    return te;
}
inline double nSG_LIN::get_dt() {
    return dt;
}
inline int nSG_LIN::N() {
    return N;
}










#endif // PDE_H_
