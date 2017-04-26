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

 
__global__ void kernel_fftshift2D(cufftDoubleComplex *IM, int im_height, int im_width);
__global__ void componentwiseMatrixMul(cufftDoubleComplex *in1, cufftDoubleComplex *in2,cufftDoubleComplex *out, int row, int col);
__global__ void zeroPadding(cufftDoubleComplex* F, cufftDoubleComplex* FP, int newCols, int newRows, int oldCols, int oldRows); 


__global__ void zeroPadding(cufftDoubleComplex* F, cufftDoubleComplex* FP, int newCols, int newRows, int oldCols, int oldRows)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = idx*newCols + idy;
    
    if(idx < newRows && idy < newCols)
    {       
        if(idx <oldRows && idy < oldCols)
        {
            FP[ind].x = F[idx*oldCols+idy].x;
        }
        else if(idx>oldRows || idy>oldCols)
        {
            FP[ind].x=0;
        }
    }
}


__global__ void componentwiseMatrixMul(cufftDoubleComplex *in1, cufftDoubleComplex *in2,cufftDoubleComplex *out,int row, int col)
{
    int indexRow=threadIdx.x + blockIdx.x*blockDim.x; 
    int indexCol=threadIdx.y + blockIdx.y*blockDim.y; 
    if(indexRow<row && indexCol<col)
    {   
        out[indexRow*col+indexCol].x = in1[indexRow*col+indexCol].x*in2[indexRow*col+indexCol].x;
        out[indexRow*col+indexCol].y = in1[indexRow*col+indexCol].y*in2[indexRow*col+indexCol].y;
    }
}


__global__ void kernel_fftshift2D(cufftDoubleComplex *IM, int imH, int imW)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = idy*imW + idx;
    int x, y, indshift;
    cufftDoubleComplex v;


    if(idx < imW && idy < imH/2)
    {
        if(idx<imW/2 && idy<imH/2)
        {
            x=idx+imW/2;
            y=idy+imH/2;
        }
        else if(idx>=imW/2 && idy<imH/2)
        {
            x=idx-imW/2;
            y=idy+imH/2;
        }

        indshift = y*imW+x;
        v.x = IM[ind].x;
        v.y = IM[ind].y;

        IM[ind].x = IM[indshift].x;
        IM[ind].y = IM[indshift].y;

        IM[indshift].x = v.x;
        IM[indshift].y = v.y;
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
    cufftDoubleComplex *IK;
    cufftHandle planZ2Z, planIZ2Z, plan2Z2Z;
    cufftResult cuError;      
    StopWatchInterface *timer=NULL;
    
    int im_height, im_width, dimK = 5, num_threads, nDevices;
    cufftDoubleComplex *im_d,*im_K;
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
    

    // Opening kernel file and output file
    FILE *fd2,*fd1;
    if((fd2=fopen(kernel_path,"r"))==NULL)
    {
        printf("Can't read kernel.\n");
    }
    if((fd1=fopen(output_file_path,"w"))==NULL)
    {
        printf("Can't open output file.\n");
    } 

    cufftDoubleComplex **A=new cufftDoubleComplex*[im_height];
    img.convertTo(img,CV_8UC1);
    for (int i=0;i<im_height;i++)
    {
        A[i] = new cufftDoubleComplex[im_width];
        for (int j=0;j<im_width;j++)
        {
            A[i][j].x=(double)img.at<uchar>(j,i);
            A[i][j].y=0;
            //printf("%lf\n",A[i][j].x);
        }
    }
    /* - - - Building the Kernel with 0-padding - - - */
    cufftDoubleComplex **K=new cufftDoubleComplex*[im_height];
    for (int i=0;i<im_height;i++)
    {
        double numk;
        K[i]=new cufftDoubleComplex[im_width];
        for (int j=0;j<im_width;j++)
        {
            if((i >= ((im_height/2) - 2)) && (i <= ((im_height/2) + 2)) && (j >=((im_width/2)-2)) && (j <=((im_width/2)+2))){
                fscanf(fd2,"%lf",&numk);
                K[i][j].x=numk;
                K[i][j].y=0.0;
            } else {
                K[i][j].x=0.0;
                K[i][j].y=0.0;
            }
        }
    }
    /* ------------------------------    Memory allocation    --------------------------------- */
    if(cudaMalloc((void**) &im_d, im_width*im_height*sizeof(cufftDoubleComplex)) != cudaSuccess){
        cout<<"Error Memory Allocation (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    if(cudaMalloc((void**) &im_K, dimK*dimK*sizeof(cufftDoubleComplex)) != cudaSuccess){
        cout<<"Error Memory Allocation (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    if(cudaMalloc((void**) &IM, im_width*im_height*sizeof(cufftDoubleComplex)) != cudaSuccess)
    {
        cout<<"Error Memory Allocation (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    if(cudaMalloc((void**) &IK, im_width*im_height*sizeof(cufftDoubleComplex)) != cudaSuccess)
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
        cudaMemcpy2D(im_d + i*im_width, sizeof(cufftDoubleComplex), A[i], sizeof(cufftDoubleComplex), sizeof(cufftDoubleComplex), im_width, cudaMemcpyHostToDevice);
    }
    
    for(int i=0; i<im_height; ++i)
    {
        cudaMemcpy2D(IK + i*im_width, sizeof(cufftDoubleComplex), K[i], sizeof(cufftDoubleComplex), sizeof(cufftDoubleComplex), im_width, cudaMemcpyHostToDevice);
    }
    
    /* Creating plans */
    cuError = cufftPlan2d(&planZ2Z, im_width, im_height, CUFFT_Z2Z);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"Error creating FFT plan (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    cuError = cufftPlan2d(&plan2Z2Z, im_width, im_height, CUFFT_Z2Z);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"Error creating FFT plan (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    cuError = cufftPlan2d(&planIZ2Z, im_width, im_height, CUFFT_Z2Z);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"Error creating IFFT plan (line "<<__LINE__<<")"<<endl;
        return 0;
    }
    
    /* - - - Fast Fourier Transform on image - - - */
    cuError = cufftExecZ2Z(planZ2Z,im_d, IM, CUFFT_FORWARD);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"Error "<<cuError<<" during executing CUFFT (line "<<__LINE__<<")"<<endl;
        return cuError;
    }
    kernel_fftshift2D<<<dimGrid, dimBlock>>>(IM, im_height, im_width);
    
    /* - - - Fast Fourier Transform on kernel - - - */
    cuError=cufftExecZ2Z(plan2Z2Z,IK, FK,CUFFT_FORWARD);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"Error "<<cuError<<" during executing CUFFT (line "<<__LINE__<<")"<<endl;
        return cuError;
    }
    kernel_fftshift2D<<<dimGrid, dimBlock>>>(FK, im_height, im_width);
    
    /* Component-wise matrix-mul */
    componentwiseMatrixMul<<<dimGrid, dimBlock>>> (IM,FK,IM, im_height, im_width);
    
    /* - - - Executing IFFT and shifting back - - - */
#ifdef DEBUG
    printf("Shifted Image\n");    
#endif    
    kernel_fftshift2D<<<dimGrid, dimBlock>>>(IM, im_height, im_width);
    
    cuError=cufftExecZ2Z(planIZ2Z, IM,im_d,CUFFT_INVERSE);
    if(cuError != CUFFT_SUCCESS)
    {
        cout<<"Error "<<cuError<<" during executing CUIFFT (line "<<__LINE__<<")"<<endl;
        return cuError;
    }
    kernel_fftshift2D<<<dimGrid, dimBlock>>>(im_d, im_height, im_width);
    
    /* - - - Generating output - - - */
    cufftDoubleComplex *c = (cufftDoubleComplex*)malloc(im_width*im_height*sizeof(cufftDoubleComplex));
    cudaMemcpy(c, im_d, sizeof(cufftDoubleComplex)*im_height*im_width , cudaMemcpyDeviceToHost);
    
    // Stopping timer and computing elapsed time
    sdkStopTimer(&timer);
    gpuTime = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    printf("Execution time %8.4f ms\n",gpuTime); //Printing elapsed time
#ifdef DEBUG    
    printf("Generating output. . .\n");
#endif    
    long double max = c[0].x;
    for (int i = 0; i < im_height; i++)
    {
        for (int j =0 ; j< im_width; j++)
        {
            fprintf(fd1,"%lf ",c[i*im_width + j].x);
            if(c[i*im_width + j].x > max)
                max = c[i*im_width + j].x;
        }
        fprintf(fd1,"\n");
    }
    img.convertTo(img,CV_64F);
    for(int i = 0; i < im_height; i++){
        for(int j =0; j < im_width; j++){
            img.at<double>(j,i) = floor((c[i*im_width + j].x/max)*255);
            //printf("%d\n",(unsigned)floor((c[i*im_width + j].x/max)*255));
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
    cufftDestroy(plan2Z2Z);
    cufftDestroy(planIZ2Z);
    cufftDestroy(planZ2Z);
    
    return 0;
}
