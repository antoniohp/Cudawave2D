#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#define NR_END 1
#define I2D(Nx, i, j) ((i) + (Nx)*(j))

double *dvector (long N)
{
	double *v;
	v = (double *)malloc(sizeof(double)*N);
	return v;
}

void zero_matrix(double *a, int Nx, int Ny)
{
	int i, j;

	for (j = 0; j < Ny; j++) {
		for (i = 0; i < Nx; i++) {
			a[I2D(Nx, i, j)] = 0.0;
		}
	}
}

void print_mat(FILE *fptr,
	double *a, int Nx, int Ny)
{
	int i, j;

	for (j = 0; j < Ny; j++) {
		for (i = 0; i < Nx; i++) {
			fprintf(fptr, "%6.3f" , a[I2D(Nx, i, j)]);
		}
		fprintf(fptr, "\n");
	}
	fclose(fptr);
}

void initialize(double *a, double *x, double *y, int Nx, int Ny)
{
	int i, j;

	for (j = 1; j < Ny-1; j++) {
		for (i = 1; i < Nx-1; i++) {
			a[I2D(Nx, i, j)] = sin(M_PI*y[i])*sin(M_PI*x[j]);
		}
	}
}
int iDivUp(int a, int b){ 
	return (((a) + (b) - 1)/(b));
}
