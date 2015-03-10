shearshear
==========

GPU shear-shear code

/* Authors: Miguel Cardenas-Montes (1)
            Ignacio Sevilla (2)
            Christopher Bonnett (3)
            Rafael Ponce (4)
            

(1): miguel.cardenas@ciemat.es
(2): ignacio.sevilla@ciemat.es
(3): c.bonnett@gmail.com
(4): rafael.ponce@ciemat.es

NOTE: This code has been successfully tested on NVIDIA GTX295, C1060, C2050, C2070 and GT555M. */

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

