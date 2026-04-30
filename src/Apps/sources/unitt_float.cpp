#include "RingComms.hpp"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int irank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &irank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Set the size of the buffers
    uint32_t bufsize = static_cast<uint32_t>(102400000);
    double bufsize_KB = static_cast<double>(bufsize * sizeof(float)) / static_cast<double>(1024);
    if (irank == 0)
    {
        printf("Buffer size: %.4f KB\n", bufsize_KB);
    }

    // Create the buffers, initialized to irank for verification
    float* sendBuf = (float*)calloc(bufsize, sizeof(float));
    float* recvBuf = (float*)calloc(bufsize, sizeof(float));
    for (uint32_t i = 0; i < bufsize; i++) {
        sendBuf[i] = static_cast<float>(irank);
    }

    for (int ir = 0; ir < nranks; ir++)
    {
        if (irank == ir) {
            printf("Rank %d sendBuf[0] = %f\n", irank, sendBuf[0]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Call the ring communication pattern
    for (uint32_t iter = 0; iter < 10; iter++) {
        RingComms<uint32_t, float>::Blocking(irank, nranks, sendBuf, recvBuf, bufsize);
    }

    for (int ir = 0; ir < nranks; ir++)
    {
        if (irank == ir) {
            printf("Rank %d sendBuf[0] = %f\n", irank, sendBuf[0]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Call the ring communication pattern
    for (uint32_t iter = 0; iter < 10; iter++)
    {
        RingComms<uint32_t, float>::NonBlocking(irank, nranks, sendBuf, recvBuf, bufsize);
    }

    for (int ir = 0; ir < nranks; ir++)
    {
        if (irank == ir)
        {
            printf("Rank %d sendBuf[0] = %f\n", irank, sendBuf[0]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Create window for RMA operations
    MPI_Win win;
    MPI_Win_create(sendBuf, bufsize * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Call the ring communication pattern
    for (uint32_t iter = 0; iter < 10; iter++)
    {
        RingComms<uint32_t, float>::PutGet(irank, nranks, sendBuf, recvBuf, bufsize, win);
    }

    for (int ir = 0; ir < nranks; ir++)
    {
        if (irank == ir)
        {
            printf("Rank %d sendBuf[0] = %f\n", irank, sendBuf[0]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Free the buffers
    free(sendBuf);
    free(recvBuf);

    MPI_Finalize();
    return 0;
}