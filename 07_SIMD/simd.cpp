#include<iostream>
#include <immintrin.h> 

using namespace std;

void function1()
{
    // Create 5 arrays of 8 integers,
    // then use AVX to vertically sum these arrays.
    int data[5][8];
    for (int i=0; i<5; i++)
    {
        for (int j=0; j<8; j++)
        {
            data[i][j] = i+1;
        }
    }

    int f[8] = {0,0,0,0,0,0,0,0};
    __m256i V = _mm256_load_si256( (const __m256i *)&f[0]);
    __m256i A;
    for (int i=0; i<5; i++)
    {
        A = _mm256_load_si256((const __m256i *)&data[i][0] );
        V = _mm256_add_epi32(V, A);
    }
    _mm256_store_si256( (__m256i *)&f[0], V);
    for (int i=0; i<8; i++)
    {
        cout << i << " " << f[i] << endl;
    }
}

void function2()
{
    /*
    Then create 2 arrays of more than 8 integers,
     and use a for loop that increments by 8 
     (the number of 32-bit integers that can fit in a 256 bit register) 
     and use AVX to vertically sum the arrays 8 elements at a time.
    */
   int data[2][32];
    for (int i=0; i<2; i++)
    {
        for (int j=0; j<32; j++)
        {
            data[i][j] = j;
        }
    }

    for (int i=0; i<32; i+=8)
    {
        __m256i A = _mm256_load_si256((const __m256i *)&data[0][i] );
        __m256i B = _mm256_load_si256((const __m256i *)&data[1][i] );
        B = _mm256_add_epi32(B, A);
        _mm256_store_si256( (__m256i *)&data[1][i], B);
    }

    for (int i=0; i<32; i++)
    {
        cout << i << " " << data[1][i] << endl;
    }
}

int main()
{
    function1();
    function2();
    return 0;
}