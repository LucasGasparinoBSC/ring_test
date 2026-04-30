/**
 * @file RingComms.hpp
 * @author your name (you@domain.com)
 * @brief Implements variations of ring communication patterns for parallel processing.
 * @version 0.1
 * @date 2026-04-30
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#ifndef RINGCOMMS_HPP
#define RINGCOMMS_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

namespace mpi_utils {
    template <typename T> MPI_Datatype MPIType();

    // Specific specializations
    template <> inline MPI_Datatype MPIType<int>() { return MPI_INT; }
    template <> inline MPI_Datatype MPIType<uint32_t>() { return MPI_UINT32_T; }
    template <> inline MPI_Datatype MPIType<uint64_t>() { return MPI_UINT64_T; }
    template <> inline MPI_Datatype MPIType<float>() { return MPI_FLOAT; }
    template <> inline MPI_Datatype MPIType<double>() { return MPI_DOUBLE; }

    // Timing utility
    template <typename FUNC>
    double timeFunction(FUNC &&f)
    {
        double t0 = MPI_Wtime();
        f();
        double t1 = MPI_Wtime();
        double tfunc = t1 - t0;
        MPI_Allreduce(MPI_IN_PLACE, &tfunc, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return tfunc;
    }
}

template<typename ITYPE, typename VTYPE>
class RingComms
{
    private:
        static void computeTargets(int &irank, int &nranks, int &leftRank, int &rightRank);
    public:
        static void Blocking(int& irank, int& nranks, VTYPE*__restrict sendBuf, VTYPE*__restrict recvBuf, ITYPE& bufSize);
        static void NonBlocking(int& irank, int& nranks, VTYPE*__restrict sendBuf, VTYPE*__restrict recvBuf, ITYPE& bufSize);
        static void PutGet(int& irank, int& nranks, VTYPE*__restrict sendBuf, VTYPE*__restrict recvBuf, ITYPE& bufSize, MPI_Win& win);
};

#endif // RINGCOMMS_HPP