#include "RingComms.hpp"

// Compute left/right targets based on rank and size
template<typename ITYPE, typename VTYPE>
void RingComms<ITYPE, VTYPE>::computeTargets(int& irank, int& nranks, int& leftRank, int& rightRank)
{
    // Compute left and right ranks using % operator
    leftRank = (irank - 1 + nranks) % nranks;
    rightRank = (irank + 1) % nranks;
}

template<typename ITYPE, typename VTYPE>
void RingComms<ITYPE, VTYPE>::Blocking(int& irank, int& nranks, VTYPE*__restrict sendBuf, VTYPE*__restrict recvBuf, ITYPE& bufSize)
{
    // timing vars
    double timeTotal, timeComms;

    // Time the full execution
    timeTotal = mpi_utils::timeFunction([&] {
        // Create and compute next targets
        int leftRank, rightRank;
        computeTargets(irank, nranks, leftRank, rightRank);

        // Left->Right SendRecv
        timeComms = mpi_utils::timeFunction([&] {
             MPI_Sendrecv(sendBuf, bufSize, mpi_utils::MPIType<VTYPE>(), leftRank, 0,
                          recvBuf, bufSize, mpi_utils::MPIType<VTYPE>(), rightRank, 0,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        });

        // Add irank to recvBuf for verification
        for (ITYPE i = 0; i < bufSize; i++) {
            recvBuf[i] += static_cast<VTYPE>(1);
        }

        // Right->Left SendRecv (reverse)
        timeComms += mpi_utils::timeFunction([&] {
            MPI_Sendrecv(recvBuf, bufSize, mpi_utils::MPIType<VTYPE>(), rightRank, 0,
                         sendBuf, bufSize, mpi_utils::MPIType<VTYPE>(), leftRank, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        });
    });

    // Print timing results
    if (irank == 0) printf("Total time: %f (ms) | Comms time: %f (ms)\n", timeTotal * 1000.0, timeComms * 1000.0);
}

template class RingComms<uint32_t, int>;
template class RingComms<uint64_t, int>;
template class RingComms<uint32_t, uint32_t>;
template class RingComms<uint64_t, uint32_t>;
template class RingComms<uint32_t, uint64_t>;
template class RingComms<uint64_t, uint64_t>;
template class RingComms<uint32_t, float>;
template class RingComms<uint64_t, float>;
template class RingComms<uint32_t, double>;
template class RingComms<uint64_t, double>;