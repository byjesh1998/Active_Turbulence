#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include "mkl.h"
#include <armadillo>
#include <ctime>
#include <csignal>



using namespace std;
using namespace arma;

extern "C" void create_descriptor_handles(int, int);
extern "C" void fft_forward(double *, MKL_Complex16 *);
extern "C" void fft_backward(MKL_Complex16 *in, double *out);
extern "C" void fft_padded_forward(double *, MKL_Complex16 *);
extern "C" void fft_padded_backward(MKL_Complex16 *in, double *out);
extern "C" void free_descriptor_handles();

void sample_noise(cx_double *z);
bool rdata(const string &input_dir);
//
// Convolution object
class Convolution {
public:
    Convolution();
    void convolve2d(Mat<cx_double> &out, Mat<cx_double> &A, Mat<cx_double> &B);
    void convolve2d_same(Mat<cx_double> &out, Mat<cx_double> &A);
    void pad_matrix(cx_double *out, cx_double *in);
    void remove_pad(cx_double *out, cx_double *in);
    void cleanup();

private:
    mat conv_out_1;
    mat conv_out_2;
    cx_mat conv_out_ft;
    cx_mat pad_ft1;
    cx_mat pad_ft2;
};

// SIGINT handler
volatile sig_atomic_t sigint_flag = 0;
void sigint_handler(int sig) {
    sigint_flag = 1;
}

typedef boost::iostreams::tee_device<ostream, ofstream> TeeDevice;
typedef boost::iostreams::stream<TeeDevice> TeeStream;
 
  //////////////////////
 /* Solver properties*/
//////////////////////
int steps;
int print_interval;
int Nx;
int Ny;
double Lx;
double Ly;
double dt;

// Parameters                   
double Diff;
double l;                        //lambda
double k;                        //kappa
double k1;                       //kappa'
double m;                        //mobility
double tem;                      //temperature
double eta;                      //viscosity
double a;                        //constants in the free energy
double b;
double zeta;                     //zeta

// Initial conditions             
double phi0;
double psi0;
double vx0;
double vy0;

int main(int argc, const char **argv) {
    clock_t start;
    typedef std::numeric_limits<double> dbl;
    double duration;
    double k1_eta;
    double sqrt_Ddt;
    double sqrt_2etatem;
    double sqrt_2mtem;
    //double epr_av = 0;
    double phi_av =0;
    int average_shift = 0;
    bool continue_average = false;
    bool average_epr = false;
    bool verbose = false;
    bool no_noise_phi = false;
    bool no_noise_psi = false;
    bool axy = false;
    string input_dir;
    char option;
    ofstream vx_file;
    ofstream vy_file;
    ofstream v_file;
    ofstream epr_file;
    ofstream epr_av_file;
    ofstream phi_av_file;
    ofstream log_file;
    ios_base::openmode epr_av_t = ios::trunc;
    ios_base::openmode phi_av_t = ios::trunc;
    TeeDevice td(cout, log_file);
    TeeStream log(td);

    // Real space fields           
    mat phi;
    mat psi;
    mat vx;
    mat vy;
    mat epr_av;

    // Set random seed
    arma_rng::set_seed_random();

    /* Parse program arguments */    //#to be edited 
    if (argc == 1) {
        // Too few input arguments
        cout << "Run with arguments: ./activeH_heun <in/out dir> -<options: c/C/a/v/n>" << endl;
        cout << "Options:" << endl;
        cout << "c: continue from files stored in in/out dir" << endl;
        cout << "C: continue from files, average EPR _AND_ continue averaging of EPR from where previous simulation stopped" << endl;
        cout << "   This flag automatically imposes flags a and c. Also requires the file epr_av.txt" << endl;
        cout << "   Provide step # of previous simulation with the flag" << endl;
        cout << "a: compute and average EPR" << endl;
        cout << "v: verbose mode, i.e. output full state (including EPR if this is computed) on every print interval" << endl;
        cout << "n: no noise in phi equation- use this rather than setting D=0" << endl;
        cout << "N: no noise in psi equation- use this rather than setting D=0" << endl;
        cout << "x: use AXY parameters" << endl;
        cout << "E.g.: ./activeH_heun some_dir/ -c -a -v" << endl;
        cout << "E.g.: ./activeH_heun some_dir/ -C 1000000" << endl;
        return 0;
    }
    else {
        // First argument is input directory
        argc--;
        argv++;

        input_dir = *argv;
        if (!rdata(input_dir)) return 0;
        else {
            log_file.open(input_dir + "log.txt", ios::trunc);
            log << "Parameters read from in_data file:" << endl;
            log << "Steps: " << steps << endl;
            log << "Print interval: " << print_interval << endl << endl;
            log << "Nx: " << Nx << endl;
            log << "Ny: " << Ny << endl;
            log << "Lx: " << Lx << endl;
            log << "Ly: " << Ly << endl;
            log << "dt: " << dt << endl << endl;
            log << "D: " << Diff << endl;
            log << "lambda: " << l << endl;
            log << "kappa: " << k << endl;
            log << "kappa1: " << k1 << endl;
            log << "m: " << m << endl;
            log << "zeta: " << zeta << endl << endl;
            log << "eta: " << eta << endl;
            log << "a: " << a << endl;
            log << "b: " << b << endl;
            log << "tem: " << tem << endl;
            log << "phi0: " << phi0 << endl;
            log << "psi0: " << psi0 << endl;
            log << "vx0: " << vx0 << endl;
            log << "vy0: " << vy0 << endl << endl;

            // If input is read successfully, initialize fields and epr  
            phi.ones(Nx,Ny); phi *= phi0;
            psi.ones(Nx,Ny); psi *=psi0;
            vx.ones(Nx,Ny); vx *= vx0;
            vy.ones(Nx,Ny); vy *= vy0;

            sqrt_Ddt = sqrt(Diff*dt);
            sqrt_2etatem=sqrt(2*eta*tem);
            sqrt_2mtem =sqrt (2*m*tem);
            k1_eta = k1/eta ;
        }

        // Following arguments contains options
        while (argc > 1) {
            argc--;
            argv++;

            if (**argv == '-') {
                option = *(*argv + 1);

                if (option == 'c') {
                    log << "Reading initial state from file..." << endl << endl;
                    if (!(phi.load(input_dir + "phi.txt") && psi.load(input_dir + "psi.txt") && vx.load(input_dir + "vx.txt") && vy.load(input_dir + "vy.txt"))) {
                        log << "Could not read initial state from file, check files exist..." << endl;
                        return 0;
                    }
                }
                else if (option == 'a') {
                    average_epr = true;
                }
                else if (option == 'v') {
                    verbose = true;
                }
                else if (option == 'n') {
                    no_noise_phi = true;
                }
                else if (option == 'N') {       //
                    no_noise_psi = true;
                }
                else if (option == 'x') {
                    log << "Running with parameters from AXY model" << endl << endl;
                    axy = true;         /*to be edited*/
                    l = 3/16;
                    k = 5/16;
                    k1 = 1;
                    zeta=0;
                    eta=1;
                    a= 1;
                    b=1;
                    m=1;
                    tem=1;
                    Diff=1;

                }

                else if (option == 'C') {
                    continue_average = true;
                    average_epr = true;
                    log << "Reading initial state from file..." << endl << endl;
                    if (!(phi.load(input_dir + "phi.txt") && psi.load(input_dir + "psi.txt") && vx.load(input_dir + "vx.txt") && vy.load(input_dir + "vy.txt"))) {
                        log << "Could not read initial state from file, check files exist..." << endl;
                        return 0;
                    }

                    if (argc > 1) {
                        argc--;
                        argv++;


                        average_shift = strtol(*argv, nullptr, 10);
                        if (average_shift == 0) {
                            log << "Invalid number of steps passed with flag -C" << endl;
                            return 0;
                        }
                        log << "Previous number of steps used in EPR average: " << average_shift << endl;
                    }

                    else {
                        log << "Flag -C requires specification of number of steps used in previous average" << endl;
                        return 0;
                    }
                    //vec epr_av_vec;
                    if (!epr_av.load(input_dir + "epr_av.txt")) {
                        log << "Could not load file epr_av.txt" << endl;
                        return 0;
                    }
                    //epr_av = epr_av_vec[epr_av_vec.n_rows-1];
                    log << "Read the following values from epr_av.txt:" << endl;
                    //log << "epr_av = " << epr_av << endl << endl;
                    epr_av_t = ios::app;
                }
                else {
                    log << "Unkown option: " << option << endl;
                    return 0;
                }
            }
            else {
                log << "Unknown program argument: " << *argv << endl;
                return 0;
            }
        }
    }

    /* Register signal handler function */
    signal(SIGINT, sigint_handler);

    // Convolution object
    Convolution conv;

    /* Initialize output files */   
    // velocity
    vx_file.open(input_dir + "vx_av.txt", ios::trunc);
    vy_file.open(input_dir + "vy_av.txt", ios::trunc);
    v_file.open(input_dir + "v_av.txt", ios::trunc);

    //vx_ft_file.open(input_dir + "vx_ft_av.txt", ios::trunc);
    //vy_ft_file.open(input_dir + "vy_ft_av.txt", ios::trunc);

    //phi
    phi_av_file.open(input_dir + "phi_av.txt", phi_av_t);

    // Set precision of output files
    vx_file.precision(dbl::max_digits10);
    vy_file.precision(dbl::max_digits10);
    v_file.precision(dbl::max_digits10);

    phi_av_file.precision(dbl::max_digits10);

    // EPR                      
    //if (average_epr) {
    //    epr_file.open(input_dir + "epr_ts.txt", epr_av_t);
    //    epr_av_file.open(input_dir + "epr_av.txt", epr_av_t);

    //    epr_file.precision(dbl::max_digits10);
    //    epr_av_file.precision(dbl::max_digits10);
    //}

    /* Fields, initialization */       
    // Fields Fourier space
    cx_mat phi_ft(Nx,Ny/2+1,fill::zeros);
    fft_forward(phi.memptr(), (MKL_Complex16 *)phi_ft.memptr());
    phi_ft.col(Ny/2).zeros();
    phi_ft.row(Nx/2).zeros();

    //mat z=randu(Nx,Ny,distr_param(-0.1,0.1));          //noise for hydrodynamic equation
    // Fourier space noise
    //cx_mat z_ft(Nx,Ny/2+1,fill::zeros);


    //z1.randn(); 
    //fft_forward(z.memptr(), (MKL_Complex16 *) z_ft.memptr());
    //z.col(Ny/2).zeros(); 
    //z.row(Nx/2).zeros();
    //phi_ft = z_ft;

    cx_mat psi_ft(Nx,Ny/2+1,fill::zeros);
    fft_forward(psi.memptr(), (MKL_Complex16 *)psi_ft.memptr());
    psi_ft.col(Ny/2).zeros();
    psi_ft.row(Nx/2).zeros();

    cx_mat vx_ft(Nx,Ny/2+1,fill::zeros);
    fft_forward(vx.memptr(), (MKL_Complex16 *)vx_ft.memptr());
    vx_ft.col(Ny/2).zeros();
    vx_ft.row(Nx/2).zeros();

    cx_mat vy_ft(Nx,Ny/2+1,fill::zeros);
    fft_forward(vy.memptr(), (MKL_Complex16 *)vy_ft.memptr());
    vy_ft.col(Ny/2).zeros();
    vy_ft.row(Nx/2).zeros();

    // EPR real space
    mat epr(Nx,Ny,fill::zeros);
    //mat epr_av(Nx,Ny,fill::zeros);

    // EPR Fourier space
    cx_mat epr_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat epr_av_ft(Nx,Ny/2+1,fill::zeros);


    //predictor phi term in fourier space
    //cx_mat phi_P_ft(Nx,Ny/2+1,fill::zeros);

    //velocity fourier space
    //cx_mat vx_D_ft(Nx,Ny/2+1,fill::zeros);
    //cx_mat vy_D_ft(Nx,Ny/2+1,fill::zeros);


    //psi fourier space  
    cx_mat psi_D_ft(Nx,Ny/2+1,fill::zeros);
    //cx_mat psi_D_P_ft(Nx,Ny/2+1,fill::zeros);
    //cx_mat psi_N_ft(Nx,Ny/2+1,fill::zeros);

    /* Spatial derivatives Fourier space, initialization */
    // phi derivatives                       
    cx_mat phi_x_ft(Nx,Ny/2+1,fill::zeros);      /*first order*/
    cx_mat phi_y_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat phi_d_ft(Nx,Ny/2+1,fill::zeros);


    //for the predictor part
    //cx_mat phi_P_y_ft(Nx,Ny/2+1,fill::zeros);    /*first order*/
    //cx_mat phi_P_x_ft(Nx,Ny/2+1,fill::zeros);
    //cx_mat phi_P_d_ft(Nx,Ny/2+1,fill::zeros);

    cx_mat phi_P_x_dsq_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat phi_P_y_dsq_ft(Nx,Ny/2+1,fill::zeros);

    cx_mat phi_P_x_dsq_phi_P_y_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat phi_P_y_dsq_phi_P_x_ft(Nx,Ny/2+1,fill::zeros);
    


    /* Nonlinearities Fourier space, initialization */    
    // kappa' nonlinearity


    //  deterministic term in the phi equation 
    cx_mat det_term_ft(Nx,Ny/2+1,fill::zeros);
    //cx_mat det_term_P_ft(Nx,Ny/2+1,fill::zeros);      /*deterministic term predictor*/


    //  cubic polarization nonlinearities
    cx_mat phi_sq_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat phi_c_ft(Nx,Ny/2+1,fill::zeros);
    mat phi_c(Nx,Ny,fill::zeros);
    cx_mat phi_c_dsq_ft(Nx,Ny/2+1,fill::zeros);


    mat v_D_phi(Nx,Ny,fill::zeros);
    mat phi_x(Nx,Ny,fill::zeros);
    mat phi_y(Nx,Ny,fill::zeros);
    //phi_P_x_dsq_phi_P_y = phi_x % phi_P_y_dsq;
    mat phi_P_x_dsq_phi_P_y(Nx,Ny,fill::zeros);
    mat phi_P_y_dsq_phi_P_x(Nx,Ny,fill::zeros);
    mat phi_P_y_dsq(Nx,Ny,fill::zeros);
    mat phi_P_x_dsq(Nx,Ny,fill::zeros);

    //velocity grad phi nonlinearity
    cx_mat vx_D_phi_x_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat vy_D_phi_y_ft(Nx,Ny/2+1,fill::zeros);

    mat psi_k_k1(Nx,Ny,fill::zeros);
    cx_mat psi_k_k1_ft(Nx,Ny/2+1,fill::zeros);

    //velocity grad phi nonlinearity for predictor part
    //cx_mat vx_D_phi_P_x_ft(Nx,Ny/2+1,fill::zeros);
    //cx_mat vy_D_phi_P_y_ft(Nx,Ny/2+1,fill::zeros);



    /* Wave vectors */
    vec qx_vec = join_cols(regspace<vec>(0,Nx/2),regspace<vec>(-Nx/2+1,-1)); /*q range from 1 to Nx, we avoid q=0 mode*/
    cx_mat qx(Nx,Ny/2+1,fill::ones);
    qx.each_col() %= complex<double>(0,1)*qx_vec*6.28/Nx;

    rowvec qy_vec = regspace<rowvec>(0,Ny/2);
    cx_mat qy(Nx,Ny/2+1,fill::ones);
    qy.each_row() %= complex<double>(0,1)*qy_vec*6.28/Ny;

    Mat<cx_double> q_sq = -(qx % qx) - (qy % qy);  
    Mat<cx_double> q_four = q_sq % q_sq;
    cx_mat epsilon(Nx,Ny/2+1,fill::ones);
    Mat<cx_double> eps = 1e-8* epsilon;
    Mat<cx_double> iq_four = 1 / ( q_four + eps );


    /*verbose statement*/
    if (verbose) {
        log << endl << "Printing initial state to file..." << endl;
        // Fourier transform
        fft_backward((MKL_Complex16 *) phi_ft.memptr(), phi.memptr());
        fft_backward((MKL_Complex16 *) psi_ft.memptr(), psi.memptr());
        fft_backward((MKL_Complex16 *) vx_ft.memptr(), vx.memptr());
        fft_backward((MKL_Complex16 *) vy_ft.memptr(), vy.memptr());

        // Print output
        phi.save(input_dir + "phi_0.txt", raw_ascii);
        psi.save(input_dir + "psi_0.txt", raw_ascii);
        vx.save(input_dir + "vx_0.txt", raw_ascii);
        vy.save(input_dir + "vy_0.txt", raw_ascii);

        if (average_epr) {
            log << endl << "Printing spatiotemporally resolved EPR density to file..." << endl << endl;
            // Fourier transform
            //fft_backward((MKL_Complex16 *) epr_ft.memptr(), epr.memptr());

            // Print output
            //epr.save(input_dir + "epr_0.txt", raw_ascii);
        }
    }




    start = std::clock();
    for (int i=1; i<=steps; i++) {
        /* Compute derivatives */        
        // Polarization
        phi_x_ft = qx % phi_ft;                 /*first order*/
        phi_y_ft = qy % phi_ft;

        fft_backward((MKL_Complex16 *) phi_x_ft.memptr(), phi_x.memptr());
        fft_backward((MKL_Complex16 *) phi_y_ft.memptr(), phi_y.memptr());


          ////////////////////////////////
         /* Calculating Nonlinearities */      
        ////////////////////////////////
    
        // cubic phi nonlinearities
        fft_backward((MKL_Complex16 *) phi_ft.memptr(), phi.memptr());
        phi_c = phi % (phi % phi );
        //conv.convolve2d_same(phi_sq_ft,phi_ft);
        //fft_forward((MKL_Complex16 *) phi_c.memptr(), phi_c_ft.memptr());
        fft_forward(phi_c.memptr(), (MKL_Complex16 *)phi_c_ft.memptr());
        //conv.convolve2d(phi_c_ft, phi_ft,phi_sq_ft);
        phi_c_dsq_ft = q_sq % phi_c_ft;


        phi_P_x_dsq_ft = -q_sq % phi_x_ft;
        phi_P_y_dsq_ft = -q_sq % phi_y_ft;
        fft_backward((MKL_Complex16 *) phi_P_x_dsq_ft.memptr(), phi_P_x_dsq.memptr());
        fft_backward((MKL_Complex16 *) phi_P_y_dsq_ft.memptr(), phi_P_y_dsq.memptr());
        phi_P_x_dsq_phi_P_y =  ( phi_x % phi_P_y_dsq - phi_y % phi_P_x_dsq );
        //phi_P_y_dsq_phi_P_x = phi_y % phi_P_x_dsq;
        //conv.convolve2d(phi_P_x_dsq_phi_P_y_ft, phi_x_ft,phi_P_y_dsq_ft);
        //conv.convolve2d(phi_P_y_dsq_phi_P_x_ft, phi_y_ft,phi_P_x_dsq_ft);
        fft_forward(phi_P_x_dsq_phi_P_y.memptr(), (MKL_Complex16 *)phi_P_x_dsq_phi_P_y_ft.memptr());
        //fft_forward(phi_P_y_dsq_phi_P_x.memptr(), (MKL_Complex16 *)phi_P_y_dsq_phi_P_x_ft.memptr());


          /////////////////////////////////////
         /*calculating psi in forurier space*/  
        /////////////////////////////////////
        psi_D_ft = iq_four % ( phi_P_x_dsq_phi_P_y_ft) ;   
        psi_D_ft *=  k1_eta ;

        psi_ft = psi_D_ft;          //deterministic part of psi
        fft_backward((MKL_Complex16 *) psi_ft.memptr(), psi.memptr());


          //////////////////////////////////
         /*calculating v in fourier space*/      
        //////////////////////////////////
        vx_ft = qy % psi_D_ft;
        vy_ft = - qx % psi_D_ft;

        fft_backward((MKL_Complex16 *) vx_ft.memptr(), vx.memptr());
        fft_backward((MKL_Complex16 *) vy_ft.memptr(), vy.memptr());
          //////////////////////////////////
         /* Calculating phi (Final step) */
        //////////////////////////////////
        //velocity grad phi nonlinearity
        v_D_phi = (vx % phi_x + vy % phi_y);
        //conv.convolve2d(vx_D_phi_x_ft,vx_ft,phi_x_ft);
        //conv.convolve2d(vy_D_phi_y_ft,vy_ft,phi_y_ft);
        fft_forward(v_D_phi.memptr(), (MKL_Complex16 *)vy_D_phi_y_ft.memptr());

        if (average_epr) {
            
            //to calculate epr
            /*k-k' nonlinearity*/
            psi_k_k1=  psi % phi_P_x_dsq_phi_P_y;
            fft_forward(psi_k_k1.memptr(), (MKL_Complex16 *)psi_k_k1_ft.memptr());
            //conv.convolve2d(psi_phi_y_dsq_phi_x_ft,psi_ft,phi_y_dsq_phi_x_ft);
            //conv.convolve2d(psi_phi_x_dsq_phi_y_ft,psi_ft,phi_x_dsq_phi_y_ft);
            psi_k_k1_ft*= (k-k1);


            // /*j nonlinearity*/
            // j_x_ft  = ( l * lambda_phi_d_sq_x_ft ) + ( zeta * zeta_phi_dsq_phi_x_ft );
            // j_x_ft -= a * m * phi_x_ft;
            // j_x_ft -= b * m * qx % phi_c_ft;
            // j_x_ft += m * k * qx % ( phi_xsq_ft + phi_ysq_ft );
            

            // j_y_ft  = ( l* lambda_phi_d_sq_y_ft ) + ( zeta * zeta_phi_dsq_phi_y_ft ) ;
            // j_y_ft -= a * m * phi_y_ft;
            // j_y_ft -= b * m * qy % phi_c_ft;
            // j_y_ft += k * m * qy % ( phi_xsq_ft + phi_ysq_ft );

            // if(!no_noise_phi){                          /*noise terms*/
            //     j_x_ft += sqrt_2mtem * ( z3_ft );
            //     j_y_ft += sqrt_2mtem * ( z2_ft );
            // }

            // conv.convolve2d(j_zeta_x_ft,j_x_ft,zeta_phi_dsq_phi_x_ft);
            // conv.convolve2d(j_zeta_y_ft,j_y_ft,zeta_phi_dsq_phi_y_ft);


            // conv.convolve2d(j_lambda_x_ft,j_x_ft,lambda_phi_d_sq_x_ft);
            // conv.convolve2d(j_lambda_y_ft,j_y_ft,lambda_phi_d_sq_y_ft);

            epr_ft = psi_k_k1_ft; //- zeta * ( j_zeta_x_ft + j_zeta_y_ft ) - l * ( j_lambda_x_ft + j_lambda_y_ft );
            epr_ft *= ( 1/tem );   /*dividing by temperature*/


            // Average EPR 1
            //epr_av = ((i+average_shift-1)*epr_av + epr_ft(0,0).real())/(i+average_shift);
            epr_av_ft = ((i+average_shift-1)*epr_av_ft + epr_ft)/(i+average_shift);
        }

        //calculatng deterministic part in the equation in fourier space
        det_term_ft  = - ( vy_D_phi_y_ft );
        det_term_ft -= a * m * q_sq % phi_ft;
        det_term_ft -=  k * ( q_four % phi_ft );
        det_term_ft -= b * phi_c_dsq_ft;


        // /*final stepping */
        // //phi_ft += phi_ft ;
        phi_ft += ( det_term_ft  ) * dt ;

        
        //phi_av = ((i+average_shift-1)*phi_av + phi_ft(0,0).real())/(i+average_shift);



        /* Print output */
        if (i % print_interval == 0) {
            duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
            log << "============ UPDATE ============" << endl;
            log << "Simulation directory: " << input_dir << endl;
            log << "Step: " << i << endl;
            if (continue_average) log << "(Step shifted by: " << average_shift << ")" << endl;
            log << "Time spent since last update: " << duration << endl;

            //log << "vx: " << vx_ft(0,0).real() << endl;
            //vx_file << vx_ft(0,0).real() << endl;

            //log << "vy: " << vy_ft(0,0).real() << endl;
            //vy_file << vy_ft(0,0).real() << endl;

            //log << "|<v>|: " << sqrt(vx_ft(0,0).real()*vx_ft(0,0).real() + vy_ft(0,0).real()*vy_ft(0,0).real()) << endl;
            //log << "sqrt(<v^2>): " << sqrt(v_sq_ft(0,0).real()) << endl;
            //v_file << sqrt(v_sq_ft(0,0).real()) << endl;

            if (verbose) {
                log << endl << "Printing state to file..." << endl;
                // Fourier transform
                fft_backward((MKL_Complex16 *) phi_ft.memptr(), phi.memptr());
                fft_backward((MKL_Complex16 *) psi_ft.memptr(), psi.memptr());
                fft_backward((MKL_Complex16 *) vx_ft.memptr(), vx.memptr());
                fft_backward((MKL_Complex16 *) vy_ft.memptr(), vy.memptr());

                // Print output
                phi.save(input_dir + "phi_" + to_string(i+49000000) + ".txt", raw_ascii);
                psi.save(input_dir + "psi_" + to_string(i+49000000) + ".txt", raw_ascii);
                vx.save(input_dir + "vx_" + to_string(i+49000000) + ".txt", raw_ascii);
                vy.save(input_dir + "vy_" + to_string(i+49000000) + ".txt", raw_ascii);

                phi_av_file << phi_av << endl;
            }
            if (average_epr) {
                log << endl << "===== EPR =====" << endl;
                //log << "Average EPR: " << epr_av << endl;
                epr_av_file << epr_av << endl;

                //log << "Instantaneous EPR (time series): " << epr_ft(0,0).real() << endl;
                //epr_file << epr_ft(0,0).real() << endl;

                if (verbose) {
                    log << endl << "Printing spatiotemporally resolved EPR density to file..." << endl;
                    // Fourier transform
                    fft_backward((MKL_Complex16 *) epr_ft.memptr(), epr.memptr());
                    fft_backward((MKL_Complex16 *) epr_av_ft.memptr(), epr_av.memptr());

                    // Print output
                    epr.save(input_dir + "epr_" + to_string(i+49000000) + ".txt", raw_ascii);
                    epr_av.save(input_dir + "epr_av_" + to_string(i+49000000) + ".txt", raw_ascii);
                }
            }
            log << "================================" << endl << endl;
            start = std::clock();
        }
        if (sigint_flag) {
            log << endl << "Received SIGINT, exiting gracefully..." << endl;
            break;
        }
    }

    /* Output final state */
    // Fourier transform final config
    fft_backward((MKL_Complex16 *) phi_ft.memptr(), phi.memptr());
    fft_backward((MKL_Complex16 *) psi_ft.memptr(), psi.memptr());
    fft_backward((MKL_Complex16 *) vx_ft.memptr(), vx.memptr());
    fft_backward((MKL_Complex16 *) vy_ft.memptr(), vy.memptr());

    // Save final state to file
    log << "Printing final state to file ..." << endl;
    phi.save(input_dir + "phi.txt", raw_ascii);
    psi.save(input_dir + "psi.txt", raw_ascii);
    vx.save(input_dir + "vx.txt", raw_ascii);
    vy.save(input_dir + "vy.txt", raw_ascii);

    

    // Save final EPR state to file
    if (average_epr) {
        fft_backward((MKL_Complex16 *) epr_ft.memptr(), epr.memptr());
        fft_backward((MKL_Complex16 *) epr_av_ft.memptr(), epr_av.memptr());

        epr.save(input_dir + "epr.txt", raw_ascii);
        epr_av.save(input_dir + "epr_av.txt", raw_ascii);
    }

    /* Close output files */
    // Polarization
    vx_file.close();
    vy_file.close();
    v_file.close();

    // Log stream
    log.close();
    log_file.close();

    // EPR
    if (average_epr) {
        epr_file.close();
        epr_av_file.close();
    }

    conv.cleanup();

    return 0;
}




void sample_noise(cx_double *z) {
    double a, b;

    for (int j=0; j<Ny/2; j++) {
        for (int i=0; i<Nx/2; i++) {
            if (j == 0) {
                if (i == 0) {
                    z[0] = randn<double>();
                }
                else {
                    a = randn<double>(); b = randn<double>();
                    z[i] = a + complex<double>(0,1)*b;
                    z[Nx-i] = a - complex<double>(0,1)*b;
                }
            }
            else {
                z[i + Nx*j] = randn<double>() + complex<double>(0,1)*randn<double>();
            }
        }
        for (int i=Nx/2+1; i<Nx; i++) {
            if (j == 0) {
                continue;
            }
            else {
                z[i + Nx*j] = randn<double>() + complex<double>(0,1)*randn<double>();
            }
        }
    }
}

Convolution::Convolution() {
    /* Create FFT descriptor handles */
    create_descriptor_handles(Nx,Ny);

    /* Field initialization */
    // Fields used for convolutions and paddings (global)
    conv_out_1.zeros(3*Nx/2,3*Ny/2);
    conv_out_2.zeros(3*Nx/2,3*Ny/2);
    conv_out_ft.zeros(3*Nx/2,3*Ny/4+1);
    pad_ft1.zeros(3*Nx/2,3*Ny/4+1);
    pad_ft2.zeros(3*Nx/2,3*Ny/4+1);
}

void Convolution::convolve2d(Mat<cx_double> &out, Mat<cx_double> &A, Mat<cx_double> &B) {
    // Pad matrices A and B
    pad_matrix(pad_ft1.memptr(),A.memptr());
    pad_matrix(pad_ft2.memptr(),B.memptr());

    // Transform to real space
    fft_padded_backward((MKL_Complex16 *) pad_ft1.memptr(), conv_out_1.memptr());
    fft_padded_backward((MKL_Complex16 *) pad_ft2.memptr(), conv_out_2.memptr());

    // Compute real space product
    conv_out_1 %= conv_out_2;

    // Transform back == convolution in Fourier space
    fft_padded_forward(conv_out_1.memptr(), (MKL_Complex16 *) conv_out_ft.memptr());

    // Shed off padding and zero out final row/column
    remove_pad(out.memptr(),conv_out_ft.memptr());
    out.col(Ny/2).zeros();
    out.row(Nx/2).zeros();
}

void Convolution::convolve2d_same(Mat<cx_double> &out, Mat<cx_double> &A) {
    // Pad matrix A
    pad_matrix(pad_ft1.memptr(),A.memptr());

    // Transform to real space
    fft_padded_backward((MKL_Complex16 *) pad_ft1.memptr(), conv_out_1.memptr());

    // Compute real space product
    conv_out_1 %= conv_out_1;

    // Transform back == convolution in Fourier space
    fft_padded_forward(conv_out_1.memptr(), (MKL_Complex16 *) conv_out_ft.memptr());

    // Shed off padding and zero out final row/column
    remove_pad(out.memptr(),conv_out_ft.memptr());

    out.col(Ny/2).zeros();
    out.row(Nx/2).zeros();
}

void Convolution::pad_matrix(cx_double *out, cx_double *in) {
    for (int i=0; i<Nx/2; i++) {
        for (int j=0; j<Ny/2; j++) {
            out[i + 3*Nx*j/2] = in[i + Nx*j];
        }
    }
    for (int i=Nx+1; i<3*Nx/2; i++) {
        for (int j=0; j<Ny/2; j++) {
            out[i + 3*Nx*j/2] = in[i-Nx/2 + Nx*j];
        }
    }
}

void Convolution::remove_pad(cx_double *out, cx_double *in) {
    for (int i=0; i<Nx/2; i++) {
        for (int j=0; j<Ny/2; j++) {
            out[i + Nx*j] = in[i + 3*Nx*j/2];
        }
    }
    for (int i=Nx+1; i<3*Nx/2; i++) {
        for (int j=0; j<Ny/2; j++) {
            out[i-Nx/2 + Nx*j] = in[i + 3*Nx*j/2];
        }
    }
}

void Convolution::cleanup() {
    // Free MKL FFT descriptor handles (calls c subroutine)
    free_descriptor_handles();
}

bool rdata(const string &input_dir) {
    ifstream input(input_dir + "in_data");
    string delimiter = "=";
    string line;
    string var;
    bool steps_b = false;
    bool print_interval_b = false;
    bool Nx_b = false;
    bool Ny_b = false;
    bool Lx_b = false;
    bool Ly_b = false;
    bool dt_b = false;
    bool Diff_b = false;
    bool l_b = false;
    bool k_b = false;
    bool k1_b = false;
    bool zeta_b = false;
    bool m_b = false;
    bool eta_b = false;
    bool tem_b = false;
    bool a_b = false;
    bool b_b = false;
    bool phi0_b = false;
    bool psi0_b = false;
    bool vx0_b = false;
    bool vy0_b = false;

    if(input.is_open()) {
        while (!input.eof()) {
            getline(input,line);
            if (line.find(delimiter) == string::npos) continue;
            else {
                var = line.substr(0, line.find(delimiter));
                var.erase(remove_if(var.begin(), var.end(), ::isspace), var.end());

                line.erase(0, line.find(delimiter) + delimiter.length());
                line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());

                if (var == "steps") {
                    steps = atoi(line.c_str());
                    steps_b = true;
                }
                else if (var == "pinterval") {
                    print_interval = atoi(line.c_str());
                    print_interval_b = true;
                }
                else if (var == "Nx") {
                    Nx = atoi(line.c_str());
                    Nx_b = true;
                }
                else if (var == "Ny") {
                    Ny = atoi(line.c_str());
                    Ny_b = true;
                }
                else if (var == "Lx") {
                    Lx = stod(line);
                    Lx_b = true;
                }
                else if (var == "Ly") {
                    Ly = stod(line);
                    Ly_b = true;
                }
                else if (var == "dt") {
                    dt = stod(line);
                    dt_b = true;
                }
                else if (var == "D") {
                    Diff = stod(line);
                    Diff_b = true;
                }
                else if (var == "lambda") {
                    l = stod(line);
                    k = l;
                    l_b = true;
                }
                else if (var == "kappa") {
                    k = stod(line);
                    k_b = true;
                }
                else if (var == "kappa1") {
                    k1 = stod(line);
                    k1_b = true;
                }
                else if (var == "zeta") {
                    zeta = stod(line);
                    zeta_b = true;
                }
                else if (var == "eta") {
                    eta = stod(line);
                    eta_b = true;
                }
                else if (var == "tem") {
                    tem = stod(line);
                    tem_b = true;
                }
                else if (var == "m") {
                    m = stod(line);
                    m_b = true;
                }
                else if (var == "a") {
                    a = stod(line);
                    a_b = true;
                }
                else if (var == "b") {
                    b = stod(line);
                    b_b = true;
                }
                else if (var == "phi0") {
                    phi0 = stod(line);
                    phi0_b = true;
                }
                else if (var == "psi0") {
                    psi0 = stod(line);
                    psi0_b = true;
                }
                else if (var == "vx0") {
                    vx0 = stod(line);
                    vx0_b = true;
                }
                else if (var == "vy0") {
                    vy0 = stod(line);
                    vy0_b = true;
                }
                
                else {
                    cout << "Unknown input parameter: " << var << endl;
                    return false;
                }
            }
        }
        if (!(steps_b && print_interval_b && Nx_b && Ny_b && Lx_b && Ly_b && dt_b && Diff_b && l_b && k_b && k1_b && zeta_b && eta_b && tem_b && m_b && a_b && b_b && phi0_b && psi0_b && vx0_b && vy0_b)) {
            cout << "Could not find all parameters..." << endl;
            return false;
        }
    }
    else return false;
    return true;
}
