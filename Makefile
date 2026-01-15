
CUDA_HOME ?= /usr/local/cuda
MPI_CXX   ?= mpicxx

# Query Open MPI for its compile/link flags
MPI_COMPILE_FLAGS := $(shell $(MPI_CXX) --showme:compile)
MPI_LINK_FLAGS    := $(shell $(MPI_CXX) --showme:link)


NVCC ?= nvcc
CXXFLAGS := -O2 -g -std=c++17 -I third_party

LDFLAGS  := -ldl
CUDART   := -L$(CUDA_HOME)/lib64 -lcudart


all: faulter observer glmark2_mpi_wrapper two_process_ptr_crash

# Compile the CUDA source; pass MPI compile flags so <mpi.h> is found
faulter.o: faulter.cu
	$(NVCC) $(NVFLAGS) -I third_party $(MPI_COMPILE_FLAGS) -c $< -o $@

# Link with MPI wrapper and CUDA runtime
faulter: faulter.o
	$(MPI_CXX) $(CXXFLAGS) $^ -o $@ $(MPI_LINK_FLAGS) $(CUDART) $(LDFLAGS)

# If observer uses MPI too, build it with the MPI wrapper
observer.o: observer.cu
	$(NVCC) $(NVFLAGS) -I third_party $(MPI_COMPILE_FLAGS) -c $< -o $@

observer: observer.o
	$(MPI_CXX) $(CXXFLAGS) $^ -o $@ $(MPI_LINK_FLAGS) $(CUDART) $(LDFLAGS)

glmark2_mpi_wrapper: glmark2_mpi_wrapper.c
	mpicc -o $@ $^

two_process_ptr_crash.o: two_process_ptr_crash.cu
	$(NVCC) $(NVFLAGS) -I third_party $(MPI_COMPILE_FLAGS) -c $< -o $@

two_process_ptr_crash: two_process_ptr_crash.o
	$(MPI_CXX) $(CXXFLAGS) $^ -o $@ $(MPI_LINK_FLAGS) $(CUDART) $(LDFLAGS)

clean:
	rm -f faulter observer *.o
	rm -f ./glmark2_mpi_wrapper
