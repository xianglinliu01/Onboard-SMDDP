#include<iostream>

using namespace std;
// Execute a CUDA kernel to multiply each element in the array by 2

__global__ void multiplyBy2(int *buffer, int n)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
    {
        buffer[tid] = 2*buffer[tid];
    }
}

void test_print(int *buffer, int max=10, int min=0)
{
    cout << "========== begin test ==========\n";
    for (int i=min; i<max; i++)
    {
        cout << i << " " << buffer[i] << endl;
    }
}

void streamTest()
{
    // set up the event timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream[4];
    for (int i=0; i<4; i++)
    {
        cudaStreamCreate( &stream[i]); 
    }
    
    // allocate 1 GB on CPU and move it to GPU
    int n = 1<<28;
    int *buffer = new int[n];
    for (int i=0; i<n; i++)
    {
        buffer[i] = i%10; // fill the buffer with 0,1,2..9
    }

    cudaEventRecord(start);
    // allocate buffer for each stream; then copy part of the CPU data to GPU
    int *d_buffer[4];
    for (int i=0; i<4; i++)
    {
        cudaMalloc( (void**) &d_buffer[i], n/4*sizeof(int) );
        cudaMemcpyAsync( d_buffer[i], buffer+i*n/4, n/4*sizeof(int), \
        cudaMemcpyHostToDevice, stream[i]);
    }

    // call the kernel, then move the data back to the GPU
    for (int i=0; i<4; i++)
    {
        multiplyBy2 <<< n/4/256, 256, 0, stream[i]>>>(d_buffer[i], n/4);
        cudaMemcpyAsync( buffer+i*n/4, d_buffer[i] , n/4*sizeof(int) ,\
        cudaMemcpyDeviceToHost, stream[i]);
        //test_print(buffer, 10+i*n/4, i*n/4);
    }

    // record the elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "the duration of the whole process using stream =" \
    << milliseconds << " ms" << endl;
}

int main()
{
    // Allocate an integer array with 1GB worth of data on 
    // CPU (hence number of integers should be 1GB/sizeof(int)) 
    // and fill it with some random values.
    int n = 1<<28;
    int *buffer = new int[n]; // 1G/4=2^28
    for (int i=0; i<n; i++)
    {
        buffer[i] = i%10; // fill the buffer with 0,1,2..9
    }
    test_print(buffer, 10);

    // Measure the timing for steps b-d using CUDA events 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate the GPU buffer
    int *d_buffer;
    cudaMalloc( (void **) &d_buffer, n*sizeof(int) );

    // Copy the CPU buffer into a GPU buffer 
    cudaEventRecord(start);
    cudaMemcpy(d_buffer, buffer, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "the duration of copying from CPU to GPU=" \
    << milliseconds << " ms" << endl;

    // run the kernel
    cudaEventRecord(start);
    multiplyBy2 <<< n/256, 256>>> (d_buffer, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "the duration of kernel computing=" \
    << milliseconds << " ms" << endl;

    // Copy from GPU to CPU
    cudaEventRecord(start);
    cudaMemcpy(buffer, d_buffer, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "the duration of copying from GPU to CPU=" \
    << milliseconds << " ms" << endl;

    // Test print
    test_print(buffer, 10);
    delete[] buffer;

    // run the stream test
    streamTest();
    return 0;
}