#include<iostream>
#include<cstdio>
#include<chrono>
#define CONSTANT 4 // CONSTANT need to be a char
using namespace std;

bool check_same(char *buffer, unsigned long size)
{
    bool result = true;
    for (unsigned long i=0; i<size; i++)
    {
        if (buffer[i] != CONSTANT)
        {
            result = false;
        }
    }
    return result;
}

int main(void)
{
    // set up buffer on CPU and d_buffer on GPU; note the use of char;
    // if you use other data type, such as int, then be careful with the cudaMemset, 
    // since it will set each BYTE into the designated number, so the final value of a int 
    // can look strange.
    const unsigned long n=1<<30;
    const unsigned long size = sizeof(char)*n;
    char *d_buffer;
    cudaMalloc( (void**)&d_buffer, size );
    cudaMemset( d_buffer, CONSTANT, size );
    
    char *buffer = new char[n];
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(buffer, d_buffer, size, cudaMemcpyDeviceToHost);
    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    printf("cudaMemcpy  elapsed time: %f sec\n", duration * 1e-6);
    printf("the data size is %d\n", size);
    printf("measured bandwidth = size/duration %f GB/s\n", size/duration * 1e-3);
    
    bool isSame = check_same(buffer, size);
    if (isSame)
    {
        printf("buffer is the same as d_buffer\n");
    }
    else
    {
        printf("buffer and d_buffer are different\n");
    }
    cudaDeviceReset();
    //cudaFree(d_buffer);
    free(buffer);
}
