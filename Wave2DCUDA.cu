#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include "dev_matrix.h"

// DIVIDE_INTO(x/y) for integers, used to determine # of blocks/warps etc.
#define DIVIDE_INTO(x,y) (((x) + (y) - 1)/(y))
// I2D to index into a linear memory space from a 2D array index pair
#define I2D(Nx, i, j) ((i) + (Nx)*(j))

// Block size in the i and j directions
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16


// Arreglo vel, den.
void velxden(int Nx, int Ny, double *avp, double *avs, double *rho, double *mu, double *l, double vp, double vs){
	int i, j, P;

	for (j = 0; j < Ny; j++) {
		for (i = 0; i < Nx; i++) {
		// find indices into linear memory for central point and neighbours
		P = I2D(Nx, i, j);

		avp[P] = vp;
		avs[P] = vs;
		rho[P] = 2000;
		mu[P]= pow(avs[P],2) *(rho[P]);
		l[P]=rho[P]*pow(avp[P],2)-(2.0*mu[P]);
		}
	}
}


// update sigxx y sigyy CPU
void siggxx_sigyy(int Ny,int Nx, double *mu, double *l,double h,double *vx,double *vy, double *sigxx, double *sigyy,double *sigxx_p, double *sigyy_p,double T){

int i,j,P,Q,R;
double vx_dx,vy_dy;


	for (j=2 ; j<Ny ; j++) {
	for (i=1 ; i<Nx-1 ; i++) {
	P = I2D(Nx, i, j); Q = I2D(Nx, i+1, j);
	R = I2D(Nx, i, j-1); 


	vx_dx=(vx[Q] - vx[P])/h;	
	vy_dy=(vy[P] - vy[R]) /h;

	sigxx[P] = sigxx_p[P] + ((l[P] +2*mu[P])*vx_dx + l[P] * vy_dy) * T;
	sigyy[P] = sigyy_p[P] + ((l[P] +2*mu[P])* vy_dy + l[P] * vx_dx) * T;
	
	}
	}

}

// update sigxy CPU
void siggxy(int Ny,int Nx, double *mu, double h,double *vx,double *vy, double *sigxy,double *sigxy_p,double T){

int i,j,P,Q,R;
double vx_dy,vy_dx;


	for (j=2 ; j<Ny-1 ; j++) {
	for (i=2 ; i<Nx ; i++) {
	P = I2D(Nx, i, j); Q = I2D(Nx, i-1, j);
	R = I2D(Nx, i, j+1); 

	vy_dx = (vy[P] - vy[Q]) / h;
	vx_dy = (vx[R] - vx[P]) / h;

	sigxy[P]= sigxy_p[P] + mu[P]*(vy_dx + vx_dy)*T;
	
	}
	}

}

//update vx CPU
void uvx(int Ny, int Nx,double *sigxx_p,double *sigxy_p,double *vx,double *vx_p,double T,double *rho,double h){
int i,j,P,R,Q;
double sigxx_dx,sigxy_dy;

	for (j=2 ; j<Ny ; j++) {
	for (i=2 ; i<Nx ; i++) {
	P = I2D(Nx, i, j); Q = I2D(Nx, i-1, j);
	R = I2D(Nx, i, j-1); 

	sigxx_dx = (sigxx_p[P] - sigxx_p[Q])/h;
	sigxy_dy = (sigxy_p[P] - sigxy_p[R])/h;

	vx[P] = vx_p[P] + (sigxx_dx + sigxy_dy) * T/rho[P];

	}
	}
}

//update vy CPU
void uvy(int Nx, int Ny, double *rho, double *sigxy_p, double *sigyy_p, double h,double *vy,double *vy_p,double T){

int i,j,P,Q,R,S;
double sigxy_dx,sigyy_dy;

	for (j=2 ; j<Ny-1 ; j++) {
	for (i=1 ; i<Nx-1 ; i++) {
	P = I2D(Nx, i, j); Q = I2D(Nx, i+1, j);
	R = I2D(Nx, i, j+1); S = I2D(Nx,i+1,j+1);

	sigxy_dx = (sigxy_p[Q] - sigxy_p[P])/h;
	sigyy_dy = (sigyy_p[R] - sigyy_p[P])/h;

	vy[P] = vy_p[P] + (sigxy_dx + sigyy_dy) * T/rho[P];
	}
	}
	
}


//sismograma GPU
__global__ void Seismogramm_dd(int Nx,int Nt, double *Seismogramm,double *vy,double iter){
int j,k,l,P,Q;


	j = blockIdx.y*(BLOCK_SIZE_Y) + threadIdx.y;
    k= blockIdx.x*(BLOCK_SIZE_X) + threadIdx.x;
    l= blockIdx.x*(BLOCK_SIZE_X) + threadIdx.x;

//P=I2D(Nt,l,j);
//Q=I2D(Nx,22,j);

if ( l ==iter && k==22 && j>=0 && j< Nx ){
P=I2D(Nt,l,j);
Q=I2D(Nx,k,j);
Seismogramm[P]=vy[Q];
}


//if ( j>=0 && j<Nx && l ==iter ){
//Seismogramm[P]=vy[Q];
//}



}

__global__ void Seismogramm_dd_new(int Nx, int Nt, double *Seismogramm, double *vy, double iter) {
	int idx, P, Q;
	idx = blockIdx.x*(BLOCK_SIZE_X)+threadIdx.x;

	//P=I2D(Nt,l,j);
	//Q=I2D(Nx,22,j);
	if (idx < Nx) {
		P = I2D(Nt, iter, idx);
		Q = I2D(Nx, 22, idx);
		Seismogramm[P] = vy[Q];
	}
}

// Arreglo vel, den GPU.
__global__ void velxden_d(int Nx, int Ny, double *avp, double *avs, double *rho, double *mu, double *l, double vp, double vs){
	int i, j, P;
	
	// find i and j indices of this thread
	i = blockIdx.x*(BLOCK_SIZE_X) + threadIdx.x;
	j = blockIdx.y*(BLOCK_SIZE_Y) + threadIdx.y;

	// find indices into linear memory for central point and neighbours
	P = I2D(Nx, i, j);

if (i >= 0 && i < Nx-1 && j >= 0 && j < Ny-1) {
	

		avp[P] = vp;
		avs[P] = vs;
		rho[P] = 2000;
		mu[P]= pow(avs[P],2) *(rho[P]);
		l[P]=rho[P]*pow(avp[P],2)-(2.0*mu[P]);
		}
	
}


//fuente GPU!
__global__ void fuente(int Nx,double amp, double a_fu, double iter, double T, double t0, double *sigxx_p, double *sigyy_p){
int i,j,P;
double src;
	
	i = blockIdx.x*(BLOCK_SIZE_X) + threadIdx.x;
	j = blockIdx.y*(BLOCK_SIZE_Y) + threadIdx.y;

	P=I2D(Nx,i,j);
	
	src=2.0*amp*a_fu*(iter*T-t0)*exp(-1.0*a_fu*pow((iter*T-t0),2));
		
	if (i == 21 && j == 100){

	sigxx_p[P] = sigxx_p[P]+ src;
	sigyy_p[P] = sigyy_p[P] +src;
}
}

// update sigxx y sigyy GPU
__global__ void siggxx_sigyy_d(int Ny,int Nx, double *mu, double *l,double h,double *vx,double *vy, double *sigxx, double *sigyy,double *sigxx_p, double *sigyy_p,double T){

int i,j,P,Q,R;
double vx_dx,vy_dy;

	// find i and j indices of this thread
	i = blockIdx.x*(BLOCK_SIZE_X) + threadIdx.x;
	j = blockIdx.y*(BLOCK_SIZE_Y) + threadIdx.y;

	P = I2D(Nx, i, j); Q = I2D(Nx, i+1, j);
	R = I2D(Nx, i, j-1); 

	
	if (i > 0 && i < Nx-2 && j > 1 && j < Ny-1) {

	
	vx_dx=(vx[Q] - vx[P])/h;	
	vy_dy=(vy[P] - vy[R]) /h;

	sigxx[P] = sigxx_p[P] + ((l[P] +2*mu[P])*vx_dx + l[P] * vy_dy) * T;
	sigyy[P] = sigyy_p[P] + ((l[P] +2*mu[P])* vy_dy + l[P] * vx_dx) * T;


	
	}
	

}

// update sigxy GPU
__global__ void siggxy_d(int Ny,int Nx, double *mu, double h,double *vx,double *vy, double *sigxy,double *sigxy_p,double T){

int i,j,P,Q,R;
double vy_dx,vx_dy;

	// find i and j indices of this thread
	i = blockIdx.x*(BLOCK_SIZE_X) + threadIdx.x;
	j = blockIdx.y*(BLOCK_SIZE_Y) + threadIdx.y;

	P = I2D(Nx, i, j); Q = I2D(Nx, i-1, j);
	R = I2D(Nx, i, j+1); 

	if (i > 1 && i < Nx-1 && j > 1 && j < Ny-2) {

	vy_dx = (vy[P] - vy[Q]) / h;
	vx_dy = (vx[R] - vx[P]) / h;

	sigxy[P]= sigxy_p[P] + mu[P]*(vy_dx + vx_dy)*T;
	
	
	}

}


//update vx GPU
__global__ void uvx_d(int Ny, int Nx,double *sigxx_p,double *sigxy_p,double *vx,double *vx_p,double T,double *rho,double h){
int i,j,P,R,Q;
double sigxx_dx,sigxy_dy;


	// find i and j indices of this thread
	i = blockIdx.x*(BLOCK_SIZE_X) + threadIdx.x;
	j = blockIdx.y*(BLOCK_SIZE_Y) + threadIdx.y;

	P = I2D(Nx, i, j); Q = I2D(Nx, i-1, j);
	R = I2D(Nx, i, j-1);

	if (i > 1 && i < Nx-1 && j > 1 && j < Ny-1) {

	
	sigxx_dx = (sigxx_p[P] - sigxx_p[Q])/h;
	sigxy_dy = (sigxy_p[P] - sigxy_p[R])/h;

	vx[P] = vx_p[P] + (sigxx_dx + sigxy_dy) * T/rho[P];

	
	}
}

//update vy GPU
__global__ void uvy_d(int Nx, int Ny, double *rho, double *sigxy_p, double *sigyy_p, double h,double *vy,double *vy_p,double T){

int i,j,P,Q,R,S;
double sigxy_dx,sigyy_dy;

// find i and j indices of this thread
	i = blockIdx.x*(BLOCK_SIZE_X) + threadIdx.x;
	j = blockIdx.y*(BLOCK_SIZE_Y) + threadIdx.y;

	P = I2D(Nx, i, j); Q = I2D(Nx, i+1, j);
	R = I2D(Nx, i, j+1); S = I2D(Nx,i+1,j+1);

	if (i > 0 && i < Nx-2 && j > 1 && j < Ny-2) {

	
	sigxy_dx = (sigxy_p[Q] - sigxy_p[P])/h;
	sigyy_dy = (sigyy_p[R] - sigyy_p[P])/h;



	vy[P] = vy_p[P] + (sigxy_dx + sigyy_dy) * T/rho[P];
	
	}
	
}




int main() 
{
	int Nx, Ny;
	double * u_d,* u_d1;
	double *vx, *vy, *sigxx, *sigyy, *sigxy,*vx_p, *vy_p, *sigxx_p, *sigyy_p, *sigxy_p,*Seismogramm;
	double f0,t0,a_fu,amp,cp,vp,vs,pi,*avp,*avs,*rho,*mu,*l;
	int i, j, iter,rec;
	double h, T,Nt;
	dim3 numBlocks, threadsPerBlock;
	int numBlocks_Nx;
	double clock_h, clock_d;
	int P,Q,J;
	double src;

	

	rec=10;
	char times[50];
	FILE *R,*RR;

	// domain size and number of timesteps (iterations)
	pi = 3.14159265358979323846264;
	Nx = 201;
	Ny =201;
	Nt = 1200;
	h = 20;
	
	

	cp=7000;
	vp=3300;
	vs=0;
	
	T = 4/Nt;
	//fuente
	f0=4.0;
	t0=1.0/f0;
	a_fu=2.0*pow(pi*f0,2);
	amp= 800000000.0/15.24;


	avp=dvector(Nx*Ny); avs=dvector(Nx*Ny);
	rho=dvector(Nx*Ny); mu=dvector(Nx*Ny); 
	l=dvector(Nx*Ny);

	vx=dvector(Nx*Ny);    vy=dvector(Nx*Ny);
	sigxx=dvector(Nx*Ny); sigyy=dvector(Nx*Ny);
	sigxy=dvector(Nx*Ny); 
	vx_p=dvector(Nx*Ny);    vy_p=dvector(Nx*Ny);
	sigxx_p=dvector(Nx*Ny); sigyy_p=dvector(Nx*Ny);
	sigxy_p=dvector(Nx*Ny); Seismogramm=dvector(Nt*Nx);
    u_d = dvector(Nx*Ny);   u_d1=dvector(Nt*Nx);

	zero_matrix(avp,Nx,Ny);
	zero_matrix(avs,Nx,Ny);
	zero_matrix(rho,Nx,Ny);
	zero_matrix(mu,Nx,Ny);
	zero_matrix(l,Nx,Ny);
	zero_matrix(vx,Nx,Ny);
	zero_matrix(vy,Nx,Ny);
	zero_matrix(sigxx,Nx,Ny);
	zero_matrix(sigyy,Nx,Ny);
	zero_matrix(sigxy,Nx,Ny);
	zero_matrix(vx_p,Nx,Ny);
	zero_matrix(vy_p,Nx,Ny);
	zero_matrix(sigxx_p,Nx,Ny);
	zero_matrix(sigyy_p,Nx,Ny);
	zero_matrix(sigxy_p,Nx,Ny);
	zero_matrix(Seismogramm,Nt,Nx);

	dev_matrix<double> avp_d(Nx,Ny); dev_matrix<double> avs_d(Nx,Ny);
	dev_matrix<double> rho_d(Nx,Ny); dev_matrix<double> mu_d(Nx,Ny); 
	dev_matrix<double> l_d(Nx,Ny);

	dev_matrix<double> vx_d(Nx,Ny);    dev_matrix<double> vy_d(Nx,Ny);
	dev_matrix<double> sigxx_d(Nx,Ny); dev_matrix<double> sigyy_d(Nx,Ny);
	dev_matrix<double> sigxy_d(Nx,Ny); 
	dev_matrix<double> vx_p_d(Nx,Ny);    dev_matrix<double> vy_p_d(Nx,Ny);
	dev_matrix<double> sigxx_p_d(Nx,Ny); dev_matrix<double> sigyy_p_d(Nx,Ny);
	dev_matrix<double> sigxy_p_d(Nx,Ny); dev_matrix<double> Seismogramm_d(Nt,Nx);  
//set arrays
	avp_d.set(avp,Nx,Ny); 
	avs_d.set(avs,Nx,Ny);
	rho_d.set(rho,Nx,Ny);
	mu_d.set(mu,Nx,Ny); 
	l_d.set(l,Nx,Ny);
	vx_d.set(vx,Nx,Ny);
	vy_d.set(vy,Nx,Ny);
	sigxx_d.set(sigxx,Nx,Ny);
	sigyy_d.set(sigyy,Nx,Ny);
	sigxy_d.set(sigxy,Nx,Ny); 
	vx_p_d.set(vx,Nx,Ny);  
	vy_p_d.set(vy,Nx,Ny);
	sigxx_p_d.set(sigxx,Nx,Ny);
	sigyy_p_d.set(sigyy,Nx,Ny);
	sigxy_p_d.set(sigxy,Nx,Ny); 
    Seismogramm_d.set(Seismogramm,Nt,Nx); 
    





	// set threads and blocks
	numBlocks = dim3(iDivUp(Nx,BLOCK_SIZE_X), iDivUp(Ny,BLOCK_SIZE_Y));
	threadsPerBlock = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	numBlocks_Nx = iDivUp(Nx, BLOCK_SIZE_X);
	



	//int Nx, int Ny,  alp,  *in,  *out,  *Avp,  *Avs,  *Rho,  *Mu,  *L,  vp,  vs, Xl, Xr,  h, *d_x, *K_x, *alpha_x, *a_x, *b_x, *d_x_2, *K_x_2, *alpha_x_2, *a_x_2, *b_x_2, T, npml, Lx, Ly, Rc, d0_x, d0_y, Xl, Xr)
	// cpu loop
	printf("CPU start!\n");
	

	clock_h = double(clock()) / CLOCKS_PER_SEC;
	velxden(Nx,Ny,avp,avs,rho, mu, l,vp,vs);

	
	for (iter = 0; iter < Nt; iter++) {
	
	P=I2D(Nx,21,100);
	src=2.0*amp*a_fu*(iter*T-t0)*exp(-1.0*a_fu*pow((iter*T-t0),2));
	
	sigxx_p[P] = sigxx_p[P]+ src;
	sigyy_p[P] = sigyy_p[P] +src;

	uvx( Ny,  Nx, sigxx_p, sigxy_p, vx, vx_p, T, rho, h);
	uvy( Nx,  Ny,  rho,  sigxy_p,  sigyy_p,  h, vy, vy_p, T);


	siggxx_sigyy(Ny,Nx,mu,l,h,vx,vy,sigxx,sigyy,sigxx_p,sigyy_p,T);
    siggxy(Ny,Nx,mu,h,vx,vy,sigxy,sigxy_p,T);

	
//---Sismograma CPU---
for (i = 0; i < Nx; i++){
Q=I2D(Nt,iter,i);
J=I2D(Nx,22,i);
Seismogramm[Q]=vx[J];
}



	vy_p=vy;
	vx_p = vx;
	sigxy_p = sigxy;
	sigxx_p = sigxx;
	sigyy_p = sigyy;




	

	}

// ---Impresión de sismograma---
  RR=fopen("sismogramaCPU","w");
	for (j=0;j<Nx-1;j++){
	for (i=0;i<Nt-1;i++){
   P=I2D(Nt,i,j);
	fprintf(RR,"%6.3f",Seismogramm[P]);
	}
	fprintf( RR, "\n");
	}
	fclose(RR);
	


	clock_h = double(clock()) / CLOCKS_PER_SEC - clock_h;
	printf("CPU end!\n");

	// gpu loop
	printf("GPU start!\n");
	clock_d = double(clock()) / CLOCKS_PER_SEC;

	velxden_d<<<numBlocks, threadsPerBlock>>>(Nx,Ny,avp_d.getData(),avs_d.getData(),rho_d.getData(),mu_d.getData(),l_d.getData(),vp,vs);

	for (iter = 0; iter < Nt; iter++) {
	
fuente<<<numBlocks, threadsPerBlock>>>(Nx,amp,a_fu,iter, T, t0,sigxx_p_d.getData(), sigyy_p_d.getData());

uvx_d<<<numBlocks, threadsPerBlock>>>(Ny,Nx,sigxx_p_d.getData(),sigxy_p_d.getData(),vx_d.getData(),vx_p_d.getData(),T,rho_d.getData(),h);

	uvy_d<<<numBlocks, threadsPerBlock>>>(Nx,Ny,rho_d.getData(),sigxy_p_d.getData(),sigyy_p_d.getData(),h,vy_d.getData(),vy_p_d.getData(),T);

	
siggxx_sigyy_d<<<numBlocks, threadsPerBlock>>>(Ny,Nx,mu_d.getData(),l_d.getData(),h,vx_d.getData(),vy_d.getData(),sigxx_d.getData(),sigyy_d.getData(),sigxx_p_d.getData(),sigyy_p_d.getData(),T);
		
	siggxy_d<<<numBlocks, threadsPerBlock>>>(Ny,Nx,mu_d.getData(),h,vx_d.getData(),vy_d.getData(),sigxy_d.getData(),sigxy_p_d.getData(),T);

//    if (iter==rec) {
//	vy_d.get(&u_d[0], Nx, Ny);
//	sprintf(times,"CVz-%d.txt",rec);
//  R=fopen(times,"w");
//	for (j=0;j<Ny-1;j++){
//	for (i=0;i<Nx-1;i++){
//	fprintf(R,"%6.3f",u_d[I2D(Nx, i, j)]);
//	}
//	fprintf( R, "\n");
//	}
//	fclose(R);
//	rec=rec+10;
//	}		

//---Kernel de sismograma GPU---
//Seismogramm_dd<<<numBlocks, threadsPerBlock>>>(Nx,Nt,Seismogramm_d.getData(),vy_d.getData(),iter);
Seismogramm_dd_new <<<numBlocks, threadsPerBlock >>>(Nx, Nt, Seismogramm_d.getData(), vx_d.getData(), iter);


	

    vy_p_d=vy_d;
	vx_p_d = vx_d;
	sigxy_p_d = sigxy_d;
	sigxx_p_d = sigxx_d;
	sigyy_p_d = sigyy_d;

	} 


//impresión de sismograma GPU (sale resultados en 0)
    Seismogramm_d.get(&u_d1[0], Nt, Nx);
    R=fopen("sismogramaGPU","w");
	for (j=0;j<Nx-1;j++){
	for (i=0;i<Nt-1;i++){
    P=I2D(Nt,i,j);
	fprintf(R,"%6.3f",u_d1[P]);
	}
	fprintf( R, "\n");
	}
	fclose(R);



	cudaThreadSynchronize();
	clock_d = double(clock()) / CLOCKS_PER_SEC - clock_d;
	printf("GPU end!\n");

	



	
	printf("CPU time = %.3fms\n",clock_h*1e3);
	printf("GPU time = %.3fms\n",clock_d*1e3);
	printf("CPU time / GPU time : %.2f\n", clock_h/clock_d);

	printf("\n");
	printf("Printing...\n");
	


	

		
	
	


	

	return 0;
}
