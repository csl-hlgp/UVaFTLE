
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include <omp.h>
#include <hip/hip_runtime.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
#include <assert.h>

#include "ftle.h"
#include "arithmetic.h"
#include "preprocess.h"

#define blockSize 512
#define Num_Streams 1



void show_GPU_devices_info(int nDevices)
{
    int i;
    hipDeviceProp_t prop;
    for (i = 0; i < nDevices; i++) {
        hipGetDeviceProperties(&prop, i);
        printf("GPU #%d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",
        prop.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",
        prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
        2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        //printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
    } 
    fflush(stdout);
}

int main(int argc, char *argv[]) {

    printf("--------------------------------------------------------\n");
    printf("|                        UVaFTLE                       |\n");
    printf("|                                                      |\n");
    printf("| Developers:                                          |\n");
    printf("|  - Rocío Carratalá-Sáez | rocio@infor.uva.es         |\n");
    printf("|  - Yuri Torres          | yuri.torres@infor.uva.es   |\n");
    printf("|  - Sergio López-Huguet  | serlohu@upv.es             |\n");
    printf("--------------------------------------------------------\n");
    fflush(stdout);

    // Check usage
    if (argc != 7)
    {
        printf("USAGE: ./executable <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval>\n");
        printf("\texecutable:    compute_ftle.\n");
        printf("\tnDim:          dimensions of the space (2D/3D).\n");
        printf("\tcoords_file:   file where mesh coordinates are stored.\n");
        printf("\tfaces_file:    file where mesh faces are stored.\n");
        printf("\tflowmap_file:  file where flowmap values are stored.\n");
        printf("\tt_eval:        time when compute ftle is desired.\n");
        printf("\tprint2file:    store result in output file.\n");    
        return 1;
    }

    int check_EOF;
    int nDevices ;
    char buffer[255];
    int nDim, nVertsPerFace, nPoints, nFaces;
    FILE *file;

    double  t_eval =  atof(argv[5]), maxTime;
    double  times[10];
    int     numThreads=0;
    double *coords;
    double *flowmap;
    int    *faces, *d2_faces;
    double *logSqrt;

    int    *nFacesPerPoint, *d2_nFacesPerPoint;
    int    *facesPerPoint,  *d2_facesPerPoint;

    int m, lda;

    // Obtain and show GPU devices information  
    printf("\nGPU devices to be used:         \n\n");    
    fflush(stdout);
    hipGetDeviceCount(&nDevices);
    show_GPU_devices_info(nDevices);
    printf("--------------------------------------------------------\n");
    fflush(stdout);

    // Initialize mesh original information  
    nDim = atoi(argv[1]);
    m = nDim; lda = nDim;
    if ( nDim == 2 ) nVertsPerFace = 3; // 2D: faces are triangles
    else 
    {
        if ( nDim == 3) nVertsPerFace = 4; // 3D: faces (volumes) are tetrahedrons
        else
        {
            printf("Wrong dimension provided (2 or 3 supported)\n");
            return 1;
        }
    }

    // Read coordinates, faces and flowmap from Python-generated files and generate corresponding GPU vectors  
    // Read coordinates information  
    printf("\nReading input data\n\n"); 
    fflush(stdout);
    printf("\tReading mesh points coordinates...        "); 
    fflush(stdout);
    file = fopen( argv[2], "r" );
    check_EOF = fscanf(file, "%s", buffer);
    if ( check_EOF == EOF )
    {
        fprintf( stderr, "Error: Unexpected EOF in read_coordinates\n" ); 
        fflush(stdout);
        exit(-1);
    }
    nPoints = atoi(buffer);
    fclose(file);
        coords = (double *) malloc ( sizeof(double) * nPoints * nDim );
        //HostAlloc( (void **)&coords, sizeof(double) * nPoints * nDim,cudaHostAllocMapped);
        //hipMalloc((void **) &coords, sizeof(double) * nPoints * nDim);
    read_coordinates(argv[2], nDim, nPoints, coords); 
    printf("DONE\n"); 
    fflush(stdout);

    // Read faces information  
    printf("\tReading mesh faces vertices...            "); 
    fflush(stdout);
    file = fopen( argv[3], "r" );
    check_EOF = fscanf(file, "%s", buffer);
    if ( check_EOF == EOF )
    {
        fprintf( stderr, "Error: Unexpected EOF in read_faces\n" ); 
        fflush(stdout);
        exit(-1);
    }
    nFaces = atoi(buffer);
    faces = (int *) malloc ( sizeof(int) * nFaces * nVertsPerFace );
            //cudaHostAlloc( (void **)&faces, sizeof(int) * nFaces * nVertsPerFace,cudaHostAllocMapped);
    read_faces(argv[3], nDim, nVertsPerFace, nFaces, faces); 
    printf("DONE\n"); 
    fflush(stdout);

    // Read flowmap information  
    printf("\tReading mesh flowmap (x, y[, z])...       "); 
    fflush(stdout);
    flowmap = (double*) malloc( sizeof(double) * nPoints * nDim ); 
            //cudaHostAlloc( (void **)&flowmap, sizeof(double) * nPoints * nDim,cudaHostAllocMapped);
    read_flowmap ( argv[4], nDim, nPoints, flowmap );
    printf("DONE\n\n"); 
    printf("--------------------------------------------------------\n"); 
    fflush(stdout);

    // Allocate additional memory at the CPU      
    //logSqrt        = (double*) malloc( sizeof(double) * nPoints);       
    nFacesPerPoint = (int *) malloc( sizeof(int) * nPoints ); // REMARK: nFacesPerPoint accumulates previous nFacesPerPoint  
#ifndef PINNED            
    logSqrt        = (double*) malloc( sizeof(double) * nPoints);       
#else
    hipHostMalloc( (void **) &logSqrt, sizeof(double) * nPoints);
#endif
            
            //cudaHostAlloc( (void **) &logSqrt, sizeof(double) * nPoints,cudaHostAllocMapped);
            //cudaHostAlloc( (void **)&nFacesPerPoint, sizeof(int) * nPoints,cudaHostAllocMapped);


    // Assign faces to vertices and generate nFacesPerPoint and facesPerPoint GPU vectors  
    create_nFacesPerPoint_vector ( nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint );
    facesPerPoint = (int *) malloc( sizeof(int) * nFacesPerPoint[ nPoints - 1 ] );

            //cudaHostAlloc( (void **)&facesPerPoint, sizeof(int) * nFacesPerPoint[ nPoints - 1 ] ,cudaHostAllocMapped);

    hipMalloc( &d2_facesPerPoint, sizeof(int)    *   nFacesPerPoint[ nPoints - 1 ]); 
    hipMalloc( &d2_faces,   sizeof(int)    * nFaces  * nVertsPerFace ); 
    hipMalloc( &d2_nFacesPerPoint, sizeof(int)    * nPoints); 
    hipMemcpy( d2_faces,   faces,   sizeof(int)    * nFaces  * nVertsPerFace, hipMemcpyHostToDevice ); 
    hipMemcpy( d2_nFacesPerPoint, nFacesPerPoint, sizeof(int) * nPoints, hipMemcpyHostToDevice );  
    //create_facesPerPoint_vector_GPU<<<(ceil(    (double)nPoints/(double)blockSize)  +1),blockSize>>> ( nDim, nPoints, nFaces, nVertsPerFace, 
        //d2_faces, d2_nFacesPerPoint, d2_facesPerPoint );
        hipLaunchKernelGGL(create_facesPerPoint_vector_GPU, (ceil(    (double)nPoints/(double)blockSize)  +1),blockSize, 0, 0, nDim, nPoints, nFaces, nVertsPerFace, 
        d2_faces, d2_nFacesPerPoint, d2_facesPerPoint );
        

    hipMemcpy( facesPerPoint,   d2_facesPerPoint,   sizeof(int)    * nFacesPerPoint[ nPoints - 1 ], hipMemcpyDeviceToHost ); 
#ifdef PINNED    
    printf("\nComputing FTLE (non pinned)...                                 ");
#else
    printf("\nComputing FTLE (pinned)...                                 ");    
#endif
    fflush(stdout);

    #pragma omp parallel default(none)  shared(stdout, logSqrt, nDim, nPoints, nFaces, nVertsPerFace, numThreads, times, faces, coords, nFacesPerPoint, facesPerPoint, flowmap, t_eval, lda, m)   //shared(sched_chunk_size, t_eval, npoints, nteval, result, mesh, d_cuda_coords_vector, d_cuda_coords_vector2, d_cuda_velocity_vector, d_cuda_velocity_vector2, d_cuda_faces_vector, d_cuda_faces_vector2, d_cuda_times_vector, d_cuda_times_vector2, nsteps_rk4, numBlocks) private(ip, it, itprev) firstprivate(multigpu)
    {
        numThreads = omp_get_num_threads();
        //hipError_t error;
        double *d_w, *d_logSqrt;
        double *d_A_ei, *d_W_ei, *d_ftl_matrix;
        double *d_gra1, *d_gra2, *d_gra3;
        double *d_cg_tensor1, *d_cg_tensor2, *d_cg_tensor3;
        double *d_coords, *d_flowmap, *d_res;
        int    *devInfo, *d_faces, *d_nFacesPerPoint, *d_facesPerPoint;

        hipSetDevice(omp_get_thread_num());

        // Allocate memory for read data at the GPU (coords, faces, flowmap) 
        //printf("Allocating memory at the GPU and copying data to it... ");
        hipMalloc( &d_coords,  sizeof(double) * nPoints * nDim );
        hipMalloc( &d_faces,   sizeof(int)    * nFaces  * nVertsPerFace );
        hipMalloc( &d_flowmap, sizeof(double) * nPoints * nDim );

        // Copy read data to GPU (coords, faces, flowmap) 
        hipMemcpy( d_coords,  coords,  sizeof(double) * nPoints * nDim,          hipMemcpyHostToDevice );
        hipMemcpy( d_faces,   faces,   sizeof(int)    * nFaces  * nVertsPerFace, hipMemcpyHostToDevice );
        hipMemcpy( d_flowmap, flowmap, sizeof(double) * nPoints * nDim,          hipMemcpyHostToDevice );

        // Allocate additional memory at the GPU 
        hipMalloc( &d_nFacesPerPoint, sizeof(int)    * nPoints);
        hipMalloc( &d_w,              sizeof(double) * nPoints * nDim );
        hipMalloc( &d_logSqrt,        sizeof(double) * nPoints);
        hipMalloc ((void**)&d_res, sizeof(double) * nPoints * nDim * nDim); 
        hipMalloc( &d_gra1, sizeof(double) * nPoints * nDim );
        hipMalloc( &d_gra2, sizeof(double) * nPoints * nDim );
        if (nDim == 3) hipMalloc( &d_gra3, sizeof(double) * nPoints * nDim );
        hipMalloc ((void**)&d_cg_tensor1, sizeof(double) * nPoints * nDim); 
        hipMalloc ((void**)&d_cg_tensor2, sizeof(double) * nPoints * nDim); 
        if (nDim == 3) hipMalloc ((void**)&d_cg_tensor3, sizeof(double) * nPoints * nDim); 
        hipMalloc ( &d_ftl_matrix, sizeof(double)*nPoints * nDim * nDim );
        hipMalloc ((void**)&d_A_ei, sizeof(double) * lda * m);            
        hipMalloc ((void**)&d_W_ei, sizeof(double) * m * nPoints);            
        hipMalloc ((void**)&devInfo, sizeof(int));

        // Copy data to GPU 
        hipMalloc( &d_facesPerPoint, sizeof(int) * nFacesPerPoint[ nPoints - 1 ]);
        hipMemcpy( d_nFacesPerPoint, nFacesPerPoint, sizeof(int) * nPoints,                       hipMemcpyHostToDevice );
        hipMemcpy( d_facesPerPoint,  facesPerPoint,  sizeof(int) * nFacesPerPoint[ nPoints - 1 ], hipMemcpyHostToDevice );

        int maxi = nFacesPerPoint[0];
        for (int i=1; i<nPoints; i++)
        {
            if ((nFacesPerPoint[i]-nFacesPerPoint[i-1])>maxi ) maxi = nFacesPerPoint[i]-nFacesPerPoint[i-1];
        }

        // Create dim3 for GPU 
        dim3 block(blockSize);
        int numBlocks = (int) (ceil(    (double)nPoints/(double)block.x)  +1);
        numBlocks = numBlocks/omp_get_num_threads() + 1;         

        dim3 grid_numCoords(numBlocks);

        struct timeval start;
        struct timeval end;                 
        double time;
        int size =   omp_get_num_threads()==1 ?   0: nPoints / omp_get_num_threads();        
        int localStride = size * omp_get_thread_num();
//#define Num_Streams 1
        hipStream_t streams[Num_Streams];
        for (int i = 0; i < Num_Streams; ++i) {hipStreamCreate(&streams[i]); }

        // Solve FTLE 

        hipDeviceSynchronize();
        gettimeofday(&start, NULL);

        // STEP 1: compute gradient, tensors and ATxA based on neighbors flowmap values 
#ifndef PINNED
        if ( nDim == 2 )
            //gpu_compute_gradient_2D <<<grid_numCoords, block>>> (localStride,
              //  nPoints, nVertsPerFace, 
                //d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, 
                //d_ftl_matrix, d_W_ei, d_logSqrt, t_eval);
                hipLaunchKernelGGL(gpu_compute_gradient_2D,grid_numCoords, block, 0, 0, localStride,
                nPoints, nVertsPerFace, 
                d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, 
                d_ftl_matrix, d_W_ei, d_logSqrt, t_eval);
        else
            //gpu_compute_gradient_3D <<<grid_numCoords, block>>> (localStride,
            //    nPoints, nVertsPerFace, 
            //    d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, 
            //    d_ftl_matrix, d_W_ei, d_logSqrt, t_eval);
                hipLaunchKernelGGL(gpu_compute_gradient_3D, grid_numCoords, block, 0, 0, localStride,
                nPoints, nVertsPerFace, 
                d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, 
                d_ftl_matrix, d_W_ei, d_logSqrt, t_eval);

        hipMemcpy (&logSqrt[localStride],  &d_logSqrt[localStride], sizeof(double) * (nPoints / omp_get_num_threads()), hipMemcpyDeviceToHost);
       
#else
        if ( nDim == 2 )
                hipLaunchKernelGGL(gpu_compute_gradient_2D,grid_numCoords, block, 0, streams[0], localStride,
                nPoints, nVertsPerFace, 
                d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, 
                d_ftl_matrix, d_W_ei, d_logSqrt, t_eval);
        else
                hipLaunchKernelGGL(gpu_compute_gradient_3D, grid_numCoords, block, 0, streams[0], localStride,
                nPoints, nVertsPerFace, 
                d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, 
                d_ftl_matrix, d_W_ei, d_logSqrt, t_eval);                

        hipMemcpyAsync (&logSqrt[localStride],  &d_logSqrt[localStride], sizeof(double) * (nPoints / omp_get_num_threads()), hipMemcpyDeviceToHost,streams[0]);
#endif


hipDeviceSynchronize();
        // Time 
        gettimeofday(&end, NULL);

        // Show time    
        time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0;

        times[omp_get_thread_num()] = time;

        //error = hipGetLastError();
       /* if ( error != cudaSuccess )
            printf("ErrCUDA A: %s\n", hipGetErrorString( error ) );*/

        fflush(stdout);
        
        // Free memory 
        hipFree(d_coords);
        hipFree(d_flowmap);
        hipFree(d_faces);
        hipFree(d_nFacesPerPoint);
        hipFree(d_facesPerPoint);
        hipFree(d_w);
        hipFree(d_logSqrt);
        hipFree(d_ftl_matrix);
        hipFree(d_gra1);
        hipFree(d_gra2);
        hipFree(d_gra3);
        hipFree(d_res);
        hipFree(d_A_ei);
        hipFree(d_W_ei);
        hipFree(d_cg_tensor1);
        hipFree(d_cg_tensor2);
        hipFree(d_cg_tensor3);
        hipFree(devInfo);

       // cudaDeviceReset();
    }
    printf("DONE\n\n");
    printf("--------------------------------------------------------\n");
    fflush(stdout);

    // Write result in output file (if desired) 
    if ( atoi(argv[6]) )
    {
        printf("\nWriting result in output file...                  ");
        fflush(stdout);
        FILE *fp_w = fopen("result_FTLE.csv", "w");
        for ( int ii = 0; ii < nPoints ; ii++ )
            fprintf(fp_w, "%f\n", logSqrt[ii]);
        fclose(fp_w);
        printf("DONE\n\n");
        printf("--------------------------------------------------------\n");
        fflush(stdout);
    }

    // Show execution time 
    maxTime = times[0];
    for ( int i = 1; i < numThreads; i++){
        if (times[i] > maxTime)
            maxTime = times[i];
    }
    printf("\nExecution time (seconds): %f\n\n", maxTime);
    printf("--------------------------------------------------------\n");
    fflush(stdout);
    
    // Free memory
    free(coords);
    free(faces);
    free(flowmap);
    //free(logSqrt);
#ifndef PINNED            
    free(logSqrt);       
#else
    hipFree(logSqrt);
#endif

    free(nFacesPerPoint);
    free(facesPerPoint);

    return 0;
}
