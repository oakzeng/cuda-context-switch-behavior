
// glmark2_mpi_wrapper.c
#include <mpi.h>
#include <unistd.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);          // satisfy the runtime
    // hand off to glmark2 (replace path if needed)
    MPI_Barrier(MPI_COMM_WORLD);
    //execlp("glmark2", "glmark2", "-b", "build:use-vbo=true:duration=1", (char*)NULL);
    execlp("glmark2", "glmark2", "-b", "ideas:use-vbo=true:duration=2", (char*)NULL);
    MPI_Finalize();
    return 0;
}

