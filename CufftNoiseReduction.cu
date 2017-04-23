#include <time.h>
#include <cmath>
#include <helper_cuda.h>
#include <helper_timer.h>
 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <stdio.h>

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv; 

#define DEBUG

enum GPU_HCS_Error
{
    GPU_HCS_SUCCESS,
    GPU_HCS_ERROR_ALLOCMEM,
    GPU_HCS_ERROR_FREEMEM,
    GPU_HCS_ERROR_CPYMEM,
    GPU_HCS_ERROR_TIME,
    GPU_HCS_ERROR_PLAN,
    GPU_HCS_ERROR_FFT
};
 
__global__ void kernel_fftshift2D(cufftDoubleComplex *IM, int im_height, int im_width);
__global__ void kernel_creat_kernel(cufftDoubleReal *IM, int im_height, int im_width);
__global__ void prodottoMatriciCompPerCompGPU(cufftDoubleComplex *in1, cufftDoubleComplex *in2,cufftDoubleComplex *out, int row, int col);
__global__ void zeroPadding(double* F, cufftDoubleReal* FP, int newCols, int newRows, int oldCols, int oldRows); 


__global__ void zeroPadding(double* F, cufftDoubleReal* FP, int newCols, int newRows, int oldCols, int oldRows)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = idx*newCols + idy;
    if(idx < newRows && idy < newCols)
    {       
        if(idx <oldRows && idy < oldCols)
        {
            FP[ind] = F[idx*oldCols+idy];
        }
        else if(idx>oldRows || idy>oldCols)
        {
            FP[ind]=0;
        }
    }
}


__global__ void prodottoMatriciCompPerCompGPU(cufftDoubleComplex *in1, cufftDoubleComplex *in2,cufftDoubleComplex *out,int row, int col)
{
    int indexRow=threadIdx.x + blockIdx.x*blockDim.x; 
    int indexCol=threadIdx.y + blockIdx.y*blockDim.y; 
    if(indexRow<row && indexCol<col)
    {   out[indexRow*col+indexCol].x=in1[indexRow*col+indexCol].x*in2[indexRow*col+indexCol].x;
        out[indexRow*col+indexCol].y=in1[indexRow*col+indexCol].y*in2[indexRow*col+indexCol].y;
    }
}


__global__ void kernel_fftshift2D(cufftDoubleComplex *IM, int im_height, int im_width)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = idy*im_width + idx;
    int x, y, indshift;
    cufftDoubleComplex v;
     
     
    if(idx < im_width && idy < im_height/2)
    {
        if(idx<im_width/2 && idy<im_height/2)
        {
            x=idx+im_width/2;
            y=idy+im_height/2;
        }
        else if(idx>=im_width/2 && idy<im_height/2)
        {
            x=idx-im_width/2;
            y=idy+im_height/2;
        }
     
        indshift = y*im_width+x;
        v.x = (double)IM[ind].x;
        v.y = (double)IM[ind].y;
     
        IM[ind].x = IM[indshift].x;
        IM[ind].y = IM[indshift].y;
     
        IM[indshift].x = v.x;
        IM[indshift].y = v.y;
    }
}
 

__global__ void kernel_shift2D(cufftDoubleReal *IM, int im_height, int im_width)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = idy*im_width + idx;
    int x, y, indshift;
    cufftDoubleReal v;


    if(idx < im_width && idy < im_height/2)
    {
        if(idx<im_width/2 && idy<im_height/2)
        {
            x=idx+im_width/2;
            y=idy+im_height/2;
        }
        else if(idx>=im_width/2 && idy<im_height/2)
        {
            x=idx-im_width/2;
            y=idy+im_height/2;
        }

        indshift = y*im_width+x;
        v = (double) IM[ind];
        IM[ind] = IM[indshift];
        IM[indshift]= v;
    }
}


__global__ void kernel_creat_kernel(cufftDoubleReal *IK, int im_height, int im_width)
{
   int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = idy*im_width + idx;
    double u, v, rad;
    double sigma1 = 1.5;
     
    if(idx < im_height && idy < im_width)
    {
        u=(idx-im_height/2)*(idx-im_height/2);
        v=(idy-im_width/2)*(idy-im_width/2);
        rad=u+v;
   
        IK[ind] = exp(-rad/(2*sigma1*sigma1));
       
    }

}

/*
 *  Image Convolution with CUFFT
 *  Correct usage : ./filter <num_threads> <output_file_path> <image_path> <kernel_path>
 */
int main(int argc, char* argv[])
{
    cufftDoubleComplex *IM;
    cufftDoubleComplex *FK;
    cufftDoubleReal *IK;
    cufftHandle planD2Z, planZ2D, plan2D2Z;
    cufftResult cuError;      
    GPU_HCS_Error reserr = GPU_HCS_SUCCESS;
    StopWatchInterface *timer=NULL;
    
    int im_height, im_width, dimK = 5, num_threads, nDevices;
    double *im_d,*im_K;
    float gpuTime;
    
    char * output_file_path, *image_path, *kernel_path;
    Mat img;
    
    if (argc < 5){
        printf("%s - correct usage: %s <num_threads> <output_file_path> <image_path> <kernel_path>\n Setting default values. . .\n",argv[0],argv[0]);    
        im_height = im_width = 512;
        num_threads = 32;
        output_file_path = (char *)"filtrata.txt";
        image_path = (char *)"512Gaussian-Noise.jpg";
        kernel_path = (char *)"Kernel51.txt";
    } else {
        image_path = argv[3];
        num_threads = atoi(argv[1]);
        output_file_path = argv[2];
        kernel_path = argv[4];
    }
    
    img = imread(image_path,0);
    if (!img.data)
    {
        printf("Could not open image \n");
        return 1;
    }
    im_width = img.cols;
    im_height = img.rows;
        
    printf("im_width:%d im_height:%d \t #threads:%d \t Output file: %s\n",im_width,im_height,num_threads,output_file_path);
    
    cudaGetDeviceCount(&nDevices);
    char maxClockDevice = 0;
    int memoryClockRate = 0;
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
#ifdef DEBUG
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Max Thread per block: %d\n\n", prop.maxThreadsPerBlock);
#endif
        if(prop.memoryClockRate > memoryClockRate){
            memoryClockRate = prop.memoryClockRate;
            maxClockDevice = i;   
        }
    }
    // Setting the fastest GPU device
    cudaSetDevice(maxClockDevice);

    printf("Set Device [%d] for Execution\n",maxClockDevice);

    dim3 dimBlock(num_threads, num_threads);
    int nbBlocsW = im_width/num_threads;
    if((im_width%num_threads) != 0)
        nbBlocsW++;
    int nbBlocsH = im_height/num_threads;
    if((im_height%num_threads) != 0)
        nbBlocsH++;
    dim3 dimGrid(nbBlocsW, nbBlocsH);
    
    dim3 dimBlockK(num_threads, num_threads);
    int nbBlocsWK = dimK/num_threads;
    if((dimK%num_threads) != 0)
        nbBlocsWK++;
    int nbBlocsHK = dimK/num_threads;
    if((dimK%num_threads) != 0)
        nbBlocsHK++;
    dim3 dimGridK(nbBlocsHK, nbBlocsWK);
    
    /* lettura dell'immagine e del kernel*/
    FILE *fd2,*fd1;
    if((fd2=fopen(kernel_path,"r"))==NULL)
    {
        printf("Can't read kernel.\n");
    }
    if((fd1=fopen(output_file_path,"w"))==NULL)
    {
        printf("Can't open output file.\n");
    } 

    double **A=new double*[im_height];
    img.convertTo(img,CV_8UC1);
    for (int i=0;i<im_height;i++)
    {
        A[i] = new double[im_width];
        for (int j=0;j<im_width;j++)
        {
            A[i][j]=img.at<uchar>(j,i);
        }
    }
    double **K=new double*[dimK];
    for (int i=0;i<dimK;i++)
    {
        double numk;
        K[i]=new double[dimK];
        for (int j=0;j<dimK;j++)
        {
            fscanf(fd2,"%lf",&numk);
            K[i][j]=numk;
        }
    }
    /* ------------------------------    Memory allocation    --------------------------------- */
    if(cudaMalloc((void**) &im_d, im_width*im_height*sizeof(double)) != cudaSuccess){
        cout<<"Error Memory Allocation (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    if(cudaMalloc((void**) &im_K, dimK*dimK*sizeof(double)) != cudaSuccess){
        cout<<"Error Memory Allocation (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    if(cudaMalloc((void**) &IM, im_width*im_height*sizeof(cufftDoubleComplex)) != cudaSuccess)
    {
        cout<<"Error Memory Allocation (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    if(cudaMalloc((void**) &IK, im_width*im_height*sizeof(cufftDoubleReal)) != cudaSuccess)
    {
        cout<<"Error Memory Allocation (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    if(cudaMalloc((void**) &FK, im_width*im_height*sizeof(cufftDoubleComplex)) != cudaSuccess)
    {
        cout<<"Error Memory Allocation (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    // Creating and starting timer
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);  

    /* --- Copying image and kernel on device --- */
    for(int i=0; i<im_height; ++i)
    {
        cudaMemcpy2D(im_d + i*im_width, sizeof(double), A[i], sizeof(double), sizeof(double), im_width, cudaMemcpyHostToDevice);
    }
    for(int i=0; i<dimK; ++i)
    {
        cudaMemcpy2D(im_K + i*dimK, sizeof(double), K[i], sizeof(double), sizeof(double), dimK, cudaMemcpyHostToDevice);
    }
    
    /* Creating plans */
    cuError = cufftPlan2d(&planD2Z, im_width, im_height, CUFFT_D2Z);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"Error creating FFT plan (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    cuError = cufftPlan2d(&plan2D2Z, im_width, im_height, CUFFT_D2Z);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"Error creating FFT plan (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    cuError = cufftPlan2d(&planZ2D, im_width, im_height, CUFFT_Z2D);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"Error creating IFFT plan (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    
    /* - - - Fast Fourier Transform on image - - - */
    cuError = cufftExecD2Z(planD2Z, (cufftDoubleReal*)im_d, IM);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"[HCS] Error "<<cuError<<" during executing CUFFT (line "<<__LINE__<<")"<<endl;
        reserr = GPU_HCS_ERROR_FFT;
        return reserr;
    }
    kernel_fftshift2D<<<dimGrid, dimBlock>>>(IM, im_height, im_width);
    zeroPadding<<<dimGrid,dimBlock>>>(im_K, IK, im_width, im_height, dimK,dimK);
    
    /* - - - Fast Fourier Transform on kernel - - - */
    cuError=cufftExecD2Z(plan2D2Z,(cufftDoubleReal*)IK, FK);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"Error "<<cuError<<" during executing CUFFT (line "<<__LINE__<<")"<<endl;
        reserr = GPU_HCS_ERROR_FFT;
        return reserr;
    }
    kernel_fftshift2D<<<dimGrid, dimBlock>>>(FK, im_height, im_width);
    
    /* Component-wise matrix-mul */
    prodottoMatriciCompPerCompGPU<<<dimGrid, dimBlock>>> (IM,FK,IM, im_height, im_width);
    
    /* - - - Executing IFFT and shifting back - - - */
    kernel_fftshift2D<<<dimGrid, dimBlock>>>(IM, im_height, im_width);
#ifdef DEBUG
    printf("Shifted Image\n");    
#endif    
    cuError=cufftExecZ2D(planZ2D, IM, (cufftDoubleReal*)im_d);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"Error "<<cuError<<" during executing CUIFFT (line "<<__LINE__<<")"<<endl;
        reserr = GPU_HCS_ERROR_FFT;
        return reserr;
    }
    
    /* - - - Generating output - - - */
    double *c = (double*) malloc(im_width*im_height*sizeof(double));
    cudaMemcpy(c, im_d, sizeof(cufftDoubleReal)*im_height*(im_width) , cudaMemcpyDeviceToHost);
    
    // Stopping timer and computing elapsed time
    sdkStopTimer(&timer);
    gpuTime = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    printf("Execution time %8.4f ms\n",gpuTime); //Printing elapsed time
#ifdef DEBUG    
    printf("Generating output. . .\n");
#endif    
    long double max = c[0];
    for (int i = 0; i < im_height; i++)
    {
        for (int j =0 ; j< im_width; j++)
        {
            fprintf(fd1,"%lf ",c[i*im_width + j]);
            if(c[i*im_width + j] > max)
                max = c[i*im_width + j];
        }
        fprintf(fd1,"\n");
    }
    img.convertTo(img,CV_64F);
    for(int i = 0; i < im_height; i++){
        for(int j =0; j < im_width; j++){
            img.at<double>(j,i) = (double) floor((c[i*im_width + j]/max)*255);
            //printf("%lf\n",floor((c[i*im_width + j]/max)*255);
        }
    }
    imwrite("output_image.jpg",img);
       
    free(c);
    free(A);
    free(K);
    cudaFree(im_d);
    cudaFree(IM);
    cudaFree(IK);
    cudaFree(FK);
    cufftDestroy(plan2D2Z);
    cufftDestroy(planD2Z);
    cufftDestroy(planZ2D);
    
    return 0;
}
