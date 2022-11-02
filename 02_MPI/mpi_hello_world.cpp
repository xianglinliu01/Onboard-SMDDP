#include<mpi.h>
#include<iostream>

using namespace std;

int main()
{
    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    cout << "Hello world from rank " << world_rank << " of total rank " << world_size << endl;

    MPI_Finalize();
}
