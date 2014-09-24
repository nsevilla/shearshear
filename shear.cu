/* Authors: Miguel Cardenas-Montes (1)
            Ignacio Sevilla (2)
            Christopher Bonnett (3)
            Rafael Ponce (4)
            

(1): miguel.cardenas@ciemat.es
(2): ignacio.sevilla@ciemat.es
(3): c.bonnett@gmail.com
(4): rafael.ponce@ciemat.es

NOTE: This code has been successfully tested on NVIDIA GTX295, C1060, C2050, C2070 and GT555M. */

               /* TO COMPILE: IT'S CLEAR THAT COMPILATION DEPENDS ON YOUR PLATFORM */

/* For Ubuntu with Cuda Libraries you have to do once:

    user@user:yourpath$ nvcc -c -arch=sm_20 GP2SSCF_GPU_v0.2.cu
    user@user:yourpath$ g++ -lcudart GP2SSCF_GPU_v0.2.o -o GP2SSCF_GPU

    NOTE: Sometimes could be necesary to do: (m value depends on your architecture, 32 bits or 64 bits)
        nvcc -c -m=32 -arch=sm_20 GP2SSCF_GPU_v0.2.cu   or
        nvcc -c -m=64 -arch=sm_20 GP2SSCF_GPU_v0.2.cu


        If you have to indicate where the libraries are you will have to do something like that:

        nvcc -arch=sm_20 GP2SSCF_GPU_v0.2.cu
        g++ -lcudart -lpthread -L/usr/local/cuda/lib64 GP2SSCF_GPU_v0.2.o -o GP2SSCF_GPU

You can also try 

    user@user:yourpath$ nvcc -arch=sm_20 -lcudart -lpthread -L/usr/local/cuda5.0/lib64  -o GP2SSCF_GPU GP2SSCF_GPU_v0.2.cu

Now, if everything went well, you just created a binary file called "GP2SSCF_GPU" */


                                            /* TO RUN */

/* This program calculates the shear-shear angular correlation function by brute force using a GPU, in the flat-sky approximation
The program receives as input parameters the following:

    Catalog: The path of your galaxy catalog, usually is a galaxy catalog for a specific redshift bin 
    Output_correlation_file: The path of your output file 

The input files must have only five columns:

    RA: Right Ascension in degrees 
    DEC: Declination also in degrees
    e1: Ellipticity 1 or Shear 1
    e2: Ellipticity 2 or Shear 2
    W: Weight corresponding to each galaxy

The output will be:

    theta: angle(degrees, in logscale)
    Xi+/DD: Xi+ divided by the number of pairs separated an angular distance theta
    Xi-/DD: Xi- divided by the number of pairs separated an angular distance theta
    Xix/DD: Xix divided by the number of pairs separated an angular distance theta
    DD: the number of pairs separated an angular distance theta
    DD_wight: the number of pairs separated an angular distance theta taking into account 
              the galaxies weight

To run the program:


In order to calculate the shear-shear correlation function you must do:


        ./GP2SSCF_GPU Catalog.txt output.txt


EXAMPLES TO CALL THE FUNCTION:


        ./GP2SSCF_GPU test.cat Shear_CF.txt


IMPORTANT: 

1.- We are not giving a test catalog but our results have been compared to the athena results, so we encourage you to try our code using the athena input file.

2.- In order to compare the athena results to our results you will take into account that the athena input data set, specifically RA and DEC, are in arcseconds, so you will have to divide both by 3600.0 in order to get them in degrees. 

3.- You can download the athena code from
 
    http://www2.iap.fr/users/kilbinge/athena/

    and the input data file from

    http://www2.iap.fr/users/kilbinge/athena/athena_1.52-testdata.tgz
*/


/* Header files */

#include <string.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include "sys/time.h"
#include <assert.h>
#include <sstream>
#include <sm_20_atomic_functions.h>
   /*#include "cuPrintf.cuh"*/

/* Constants */

#define pi 3.14159265358979323846
#define threads 32  /* It's the number of threads we are going to use per block on the GPU */
#define theta_min 0.016667 /*in degrees (use float notation)*/
#define theta_max 8.3333  /* in degrees (use float notation)*/

using namespace std;


/* Kernel */

__global__ void binning_shear(double *xd, double *yd, double *zd, double *e1, double *e2, double *weight, float *ZZ_angle, float *ZZ_xix, float *ZZ_xip, float *ZZ_xim, float *ZZ_weight, int lines_number)
{
    /* defining variables in shared memory*/
    __shared__ unsigned int dibin[threads];
    __shared__ float temp_angle[threads];
    __shared__ float temp_xix[threads];
    __shared__ float temp_xip[threads];  
    __shared__ float temp_xim[threads];
    __shared__ float temp_weight[threads];
    /*********************************/
    /* using flat-sky approximation */
    /*__shared__ double phi[threads];
    __shared__ double cosphi[threads];
    __shared__ double cos2phi[threads];
    __shared__ double cos4phi[threads];
    __shared__ double sinphi[threads];
    __shared__ double sin4phi[threads];*/
    /*********************************/
    /* using curved-sky approximation */
    __shared__ double cosphi1[threads];
    __shared__ double cos2phi1[threads];
    __shared__ double sinphi1[threads];
    __shared__ double sin2phi1[threads];
    __shared__ double cosphi2[threads];
    __shared__ double cos2phi2[threads];
    __shared__ double sinphi2[threads];
    __shared__ double sin2phi2[threads];
    __shared__ double cos2Deltaphi_p[threads];
    __shared__ double sin2Deltaphi_p[threads];
    __shared__ double cos2Deltaphi_m[threads];
    __shared__ double sin2Deltaphi_m[threads];

    /* defining in registers*/
    double x, y, z;
    double ee1, ee2;
    double xx,yy,zz;
    double eee1, eee2;
    double g11,g22,g21,g12;
    double deltax,deltaxx;
    double sinalphax,sinalphaxx;
    double cosalphax,cosalphaxx;
    double sindeltax,sindeltaxx;
    double cosdeltax,cosdeltaxx;
    /*********************************/
    /* using curved-sky approximation */
    double sinDeltaalpha,cosDeltaalpha;  /*introduced for curved sky implementation*/
    double distance, sind; /*introduced distance for curved sky implementation*/

    int cacheIndex = threadIdx.x;
  
    /* zeroing */
    temp_angle[cacheIndex]=0.0;
    temp_xix[cacheIndex]=0.0;
    temp_xim[cacheIndex]=0.0;
    temp_xip[cacheIndex]=0.0;
    temp_weight[cacheIndex]=0.0;
    /*********************************/
    /* using flat-sky approximation */
    /*phi[cacheIndex]=0.0;
    sinphi[cacheIndex]=0.0;
    cosphi[cacheIndex]=0.0;
    cos2phi[cacheIndex]=0.0;
    cos4phi[cacheIndex]=0.0;
    sin4phi[cacheIndex]=0.0;*/
    /*********************************/
    /* using full-sky */
    cosphi1[cacheIndex]=0.0;
    cos2phi1[cacheIndex]=0.0;
    sinphi1[cacheIndex]=0.0;
    sin2phi1[cacheIndex]=0.0;
    cosphi2[cacheIndex]=0.0;
    cos2phi2[cacheIndex]=0.0;
    sinphi2[cacheIndex]=0.0;
    sin2phi2[cacheIndex]=0.0;
    cos2Deltaphi_p[cacheIndex]=0.0;
    sin2Deltaphi_p[cacheIndex]=0.0;
    cos2Deltaphi_m[cacheIndex]=0.0;
    sin2Deltaphi_m[cacheIndex]=0.0;
    
    /* calculating */
    for (int i=0; i<lines_number; i++)
    {
        int dim_idx =  blockIdx.x * blockDim.x + threadIdx.x;
        x = xd[i]; //no normalization needed as, by definition, these were created on unit sphere
        y = yd[i]; //no normalization needed as, by definition, these were created on unit sphere
        z = zd[i]; //no normalization needed as, by definition, these were created on unit sphere
        ee1 = e1[i];
        ee2 = e2[i];
        sindeltax = z;
        deltax = asin(sindeltax);
        cosdeltax = cos(deltax);
        cosalphax = x/cosdeltax;
        sinalphax = y/cosdeltax;

        while (dim_idx < lines_number)
        {
            xx = xd[dim_idx];
            yy = yd[dim_idx];
            zz = zd[dim_idx];
            eee1 = e1[dim_idx];
            eee2 = e2[dim_idx];
            sindeltaxx = zz;
            deltaxx = asin(sindeltaxx);
            cosdeltaxx = cos(deltaxx);
            cosalphaxx = xx/cosdeltaxx;
            sinalphaxx = yy/cosdeltaxx;
            distance = acos( x * xx + y * yy + z * zz ); /* in radians */
            /*distance = (cosalphax*cosalphaxx + sinalphax*sinalphaxx)*cosdeltax*cosdeltaxx + sindeltax*sindeltaxx;*/
            /*distance = acos(distance);*/

            dibin[cacheIndex] = int( (log(distance * 180. /pi ) - log(theta_min)) * double(threads) / (log(theta_max)-log(theta_min)) );
      
            __syncthreads();
            if(dibin[cacheIndex]>0 && dibin[cacheIndex]<threads )
            {
                /* binning angle */
                atomicAdd( &temp_angle[dibin[cacheIndex]], 1);

                g11 = ee1*eee1;
                g22 = ee2*eee2;
                g12 = ee1*eee2;
                g21 = ee2*eee1;

                /*********************************/
                /* using flat-sky approximation */
                /*phi[cacheIndex] = atan2((sinalphax * cosalphaxx
                    - cosalphax * sinalphaxx) * cosdeltaxx,
                   cosdeltax * sindeltaxx
                   - sindeltax * cosdeltaxx
                   *(cosalphax * cosalphaxx
                   + sinalphax * sinalphaxx)) + pi/2.0;

                cosphi[cacheIndex] = cos(phi[cacheIndex]);
                sinphi[cacheIndex] = sin(phi[cacheIndex]);
                
                cos2phi[cacheIndex] = 2.* cosphi[cacheIndex] * cosphi[cacheIndex] - 1.;
                cos4phi[cacheIndex] = 2.* cos2phi[cacheIndex] * cos2phi[cacheIndex] - 1.;
                sin4phi[cacheIndex] = 4.* sinphi[cacheIndex] * cosphi[cacheIndex] * cos2phi[cacheIndex];

                atomicAdd( &temp_xip[dibin[cacheIndex]], g11 + g22 );
                atomicAdd( &temp_xim[dibin[cacheIndex]], (g11-g22)*cos4phi[cacheIndex] + (g12 + g21)* sin4phi[cacheIndex] );
                atomicAdd( &temp_xix[dibin[cacheIndex]], 0.5*((-g11+g22) * sin4phi[cacheIndex] +  (g12+g21) * cos4phi[cacheIndex]));*/

 
                /*********************************/
                /* using curved-sky */
                sind = sin(distance);
                cosDeltaalpha = cosalphax*cosalphaxx + sinalphax*sinalphaxx;
                sinDeltaalpha = cosalphax*sinalphaxx - sinalphax*cosalphaxx;
                
                cosphi1[cacheIndex] = sinDeltaalpha * cosdeltaxx / sind;
                sinphi1[cacheIndex] = (cosdeltax*sindeltaxx - sindeltax * cosdeltaxx * cosDeltaalpha) / sind;        
                cos2phi1[cacheIndex] = 2.* cosphi1[cacheIndex] * cosphi1[cacheIndex] - 1.;
                sin2phi1[cacheIndex] = 2.* cosphi1[cacheIndex] * sinphi1[cacheIndex];      

                cosphi2[cacheIndex] = -sinDeltaalpha * cosdeltax / sind;
                sinphi2[cacheIndex] = (cosdeltaxx*sindeltax - sindeltaxx * cosdeltax * cosDeltaalpha) / sind;        
                cos2phi2[cacheIndex] = 2.* cosphi2[cacheIndex] * cosphi2[cacheIndex] - 1.;
                sin2phi2[cacheIndex] = 2.* cosphi2[cacheIndex] * sinphi2[cacheIndex];      

                cos2Deltaphi_p[cacheIndex] = cos2phi1[cacheIndex] * cos2phi2[cacheIndex] + sin2phi1[cacheIndex] * sin2phi2[cacheIndex];
                sin2Deltaphi_p[cacheIndex] = sin2phi1[cacheIndex] * cos2phi2[cacheIndex] - cos2phi1[cacheIndex] * sin2phi2[cacheIndex];
                cos2Deltaphi_m[cacheIndex] = cos2phi1[cacheIndex] * cos2phi2[cacheIndex] - sin2phi1[cacheIndex] * sin2phi2[cacheIndex];
                sin2Deltaphi_m[cacheIndex] = sin2phi1[cacheIndex] * cos2phi2[cacheIndex] + cos2phi1[cacheIndex] * sin2phi2[cacheIndex];

                atomicAdd( &temp_xip[dibin[cacheIndex]], (g11 + g22) * cos2Deltaphi_p[cacheIndex] + (g12 - g21) * sin2Deltaphi_p[cacheIndex]);
                atomicAdd( &temp_xim[dibin[cacheIndex]], (g11 - g22) * cos2Deltaphi_m[cacheIndex] + (g12 + g21) * sin2Deltaphi_m[cacheIndex] );
                atomicAdd( &temp_xix[dibin[cacheIndex]], 0.5*((-g11 + g22) * sin2Deltaphi_m[cacheIndex] +  (g12 + g21) * cos2Deltaphi_m[cacheIndex]));

                /* bining weight */
                atomicAdd( &temp_weight[dibin[cacheIndex]], weight[i]*weight[dim_idx] ) ;       
            }
            dim_idx += blockDim.x * gridDim.x;
            __syncthreads();
        }
    }
  
    atomicAdd( &ZZ_angle[threadIdx.x] , temp_angle[threadIdx.x]);
    atomicAdd( &ZZ_xip[threadIdx.x] , temp_xip[threadIdx.x]);
    atomicAdd( &ZZ_xim[threadIdx.x] , temp_xim[threadIdx.x]);
    /*atomicAdd( &ZZ_xix[threadIdx.x] , temp_xix[threadIdx.x]);*/
    atomicAdd( &ZZ_weight[threadIdx.x] , temp_weight[threadIdx.x]);
}

/* This function counts the number of columns in a file */
/* NOTE that this is done only checking the first row */

int cols_number(char *input_file)
{

    /* Definition of Variables */

    char row[650];
    char * pch;
    int columns=0;

    /* Opening the input file */

    std::ifstream f_in(input_file, std::ios::in);

    /* Reading the first line */

    f_in.getline(row,650,'\n');

    /* Closing file */

    f_in.close();

    /* Counting columns */

    pch = strtok (row,"'\t'' '");
    while (pch != NULL)
    {
        columns++;
        pch = strtok (NULL, "'\t'' '");
    }
    return (columns);
}


/* This function counts the number of rows in a file */

int counting_lines(char *input_file)
{

    /* Definition of variables */

    int lines=0;
    char line[650];

    /* Opening the input file */

    std::ifstream f_in;
    f_in.open(input_file, std::ios::in);

    /* Counting lines */

    while(!f_in.eof())
    {
        if(f_in.getline(line, 650, '\n' ) != NULL)
        {
            lines=lines+1;
        }
    }
    lines=lines;

    /* Closing file */

    f_in.close();
    
    return(lines);
}


/* This function checks the input data */

int verification(int argc, char *argv[])
{

    /* Definition of variables */
   
    int columns;

    /* Opening, checking and closing the first input file */

    std::ifstream fileinputdata;
 
    fileinputdata.open(argv[1], std::ios::in);
    if (fileinputdata == NULL)
    {
        printf("Error opening data file \n");
        return(0);
    }
    fileinputdata.close();
    

    /* Checking other input parameters */


    if (argv[2]==NULL)
    {
        printf("You must introduce a name for the output file \n");
        return(0);
    }

    /* Checking cols number in every input file */

    columns=cols_number(argv[1]);
    if (columns != 5 )
    {
        printf("Number of columns in data file must be exactly 5 and the first one has to be the right ascension and the second one the declination, both in degrees, the thrid one and the fourth one must be the shear 1 and the shear 2 respectively and the last one has to be the corresponding weight \n");
        return(0);
    }
    

    return(1);
}


/* Equatorial to cartesian conversion */ 

int eq2cart_read(char *filename, int nlines, double *xd, double *yd, double *zd, double *e1, double *e2, double *weight){
        std::ifstream infile(filename);
        int n;
        double ra, dec, phi, cth, th;
        double e1_aux, e2_aux, weight_aux;

        /* We will store the data in cartesian coordinates, that's why we convert the equatorial coordinates to spherical coordinates and then to cartesian coordinates */

        for (n=0;n<nlines;n++)
        {
            infile>>ra>>dec>>e1_aux>>e2_aux>>weight; /* reading equatorial coordinates */
            phi=ra*M_PI/180.0;
            cth=cos((90.0-dec)*M_PI/180.0);
            th=acos(cth);

            /* to cartesian coordinates */
            
            xd[n]=cos(phi)*sin(th);
            yd[n]=sin(phi)*sin(th);
            zd[n]=cth;
            
            /* Storing other input data */

            e1[n]=e1_aux;
            e2[n]=e2_aux;
            weight[n]=weight_aux;
        }

        infile.close();
        
        return(0);
}


int copy2gpu(double *gpu_xd, double *gpu_yd, double *gpu_zd, double *xd, double *yd, double *zd, double *gpu_e1, double *gpu_e2, double *e1, double *e2, double *gpu_weight, double *weight, int lines_number, float *gpu_DD, float *DD, float *gpu_DD_weight, float *DD_weight, float *gpu_xip, float *gpu_xim, float *gpu_xix, float *xip, float *xim, float *xix, dim3 dimGrid){
    /* We copy the data to the GPU */  

    cudaMemcpy(gpu_xd,xd,lines_number*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_yd,yd,lines_number*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_zd,zd,lines_number*sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_e1, e1, lines_number*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_e2, e2, lines_number*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_weight, weight, lines_number*sizeof(double),cudaMemcpyHostToDevice);
  
    cudaMemcpy(gpu_DD, DD, threads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_DD_weight, DD_weight, threads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_xip, xip, threads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_xim, xim, threads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_xix, xix, threads*sizeof(float), cudaMemcpyHostToDevice);

    /* We call the needed kernel to calculate the number of pairs */
    printf("Calling kernel binning_shear\n");
    /*printf("%d %d\n",lines_number*sizeof(double),threads*sizeof(float));*/

    binning_shear <<< dimGrid, threads >>> (gpu_xd, gpu_yd, gpu_zd, gpu_e1, gpu_e2, gpu_weight, gpu_DD, gpu_xix, gpu_xip, gpu_xim, gpu_DD_weight, lines_number);

    /* We recover the results */

    cudaMemcpy(DD, gpu_DD, threads*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(DD_weight, gpu_DD_weight, threads*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xip, gpu_xip, threads*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xim, gpu_xim, threads*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xix, gpu_xix, threads*sizeof(float), cudaMemcpyDeviceToHost);
   
    return(0);
}

/* Main Function*/

int main(int argc, char *argv[])
{
  
    /* Checking if the input files and call to script meet the requirements */

    if (verification(argc, argv)==0)
    {
        exit(1);
    }

    printf("VERIFIED!\n");
    
    /* Definition of variables */

    char *input_file, *output_file;

    int lines_number;

    double *xd,*yd,*zd;
    double *e1, *e2, *weight;
    
    double *gpu_xd, *gpu_yd, *gpu_zd;
    double *gpu_e1, *gpu_e2, *gpu_weight;

    /* Assignment of Variables with inputs */

    input_file=argv[1];
    output_file=argv[2];

    /*Fixing Cuda Device */
    cudaSetDevice(2);

    /* Counting lines in every input file */
        
    lines_number=counting_lines(input_file);
  
    /* We define variables to store the input data */
    printf("Allocating variables for size:%d\n",lines_number);

    xd = (double *)malloc(lines_number * sizeof (double));
    yd = (double *)malloc(lines_number * sizeof (double));
    zd = (double *)malloc(lines_number * sizeof (double));

    e1 = (double *)malloc(lines_number * sizeof (double));
    e2 = (double *)malloc(lines_number * sizeof (double));
    weight = (double *)malloc(lines_number * sizeof (double));

    /* Opening and reading the input file */        

    printf("Transform to cartesian\n");
    eq2cart_read(input_file,lines_number,xd,yd,zd,e1,e2,weight);

    /* We define variables to send to the GPU */

    /* For input data */

    printf("cudaAllocating variables for size:%d\n",lines_number);
    cudaMalloc( (void**)&gpu_xd,lines_number * sizeof(double));
    cudaMalloc( (void**)&gpu_yd,lines_number * sizeof(double));
    cudaMalloc( (void**)&gpu_zd,lines_number * sizeof(double));

    cudaMalloc( (void**)&gpu_e1, lines_number*sizeof(double));
    cudaMalloc( (void**)&gpu_e2, lines_number*sizeof(double));
    cudaMalloc( (void**)&gpu_weight, lines_number*sizeof(double));
  
    /* We define variables to store the data (DD) */
  
    /* on CPU */

    float *DD;
    float *DD_weight;
    float *xip;
    float *xim;
    float *xix;
  
    DD = (float *)malloc(threads*sizeof(float));
    DD_weight = (float *)malloc(threads*sizeof(float));
    xip = (float *)malloc(threads*sizeof(float));
    xim = (float *)malloc(threads*sizeof(float));
    xix = (float *)malloc(threads*sizeof(float));

    for (int i=0; i<threads; i++)
    {
       DD[i] = 0.0;
       DD_weight[i] = 0.0;
       xim[i] = 0.0;
       xip[i] = 0.0;
       xix[i] = 0.0;
    }

    /* on GPU */

    float *gpu_DD;
    float *gpu_DD_weight;
    float *gpu_xip;
    float *gpu_xim;
    float *gpu_xix;
    
    cudaMalloc( (void**)&gpu_DD, threads*sizeof(float));
    cudaMalloc( (void**)&gpu_DD_weight, threads*sizeof(float));
    cudaMalloc( (void**)&gpu_xip, threads*sizeof(float));
    cudaMalloc( (void**)&gpu_xim, threads*sizeof(float));
    cudaMalloc( (void**)&gpu_xix, threads*sizeof(float));
  
    /* We define the GPU-GRID size, it's really the number of blocks we are going to use on the GPU */

    dim3 dimGrid((lines_number/threads)+1);
  
    printf("copying to gpu\n");
    copy2gpu(gpu_xd, gpu_yd, gpu_zd, xd, yd, zd, gpu_e1, gpu_e2, e1, e2, gpu_weight, weight, lines_number, gpu_DD, DD, gpu_DD_weight, DD_weight, gpu_xip, gpu_xim, gpu_xix, xip, xim, xix, dimGrid);

    /* Opening the output file */		

    std::ofstream f_out(output_file);

    /*for (int i=1;i<threads;i++) */
    for (int i=0;i<threads;i++) 
    {
      f_out<<60*exp((i+0.5)/threads*(log(theta_max)-log(theta_min))+log(theta_min))<<"\t"<<xip[i]/DD[i]<<"\t"<<xim[i]/DD[i]<<"\t"<<xix[i]/DD[i]<<"\t"<<DD[i]<<"\t"<<DD_weight[i]<<endl;        
    }
  
    /* Closing output files */

    f_out.close();

    /* Freeing memory on the GPU */

    cudaFree(gpu_xd);
    cudaFree(gpu_yd);
    cudaFree(gpu_zd);
    cudaFree(gpu_e1);
    cudaFree(gpu_e2);
    cudaFree(gpu_weight);

    cudaFree(gpu_DD);
    cudaFree(gpu_DD_weight);
    cudaFree(gpu_xip);
    cudaFree(gpu_xim);
    cudaFree(gpu_xix);

    /* Freeing memory on the CPU */

    free(xd);
    free(yd);
    free(zd);
    free(e1);
    free(e2);
    free(weight);

    free(DD);
    free(DD_weight);
    free(xip);
    free(xim);
    free(xix);
    
    return(0);
}
