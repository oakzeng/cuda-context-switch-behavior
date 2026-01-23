
// glmark2_mpi_wrapper.c
#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char** argv) {
    int duration = 1;
    if (argc > 1) duration = atoi(argv[1]);
    MPI_Init(&argc, &argv);          // satisfy the runtime
    printf("glmark2 PID %d\n", getpid());
    printf("glmark2 duration %d\n", duration);
    char bench_arg[128];
    snprintf(bench_arg, sizeof(bench_arg), "build:use-vbo=true:duration=%d", duration);
    // hand off to glmark2 (replace path if needed)
    MPI_Barrier(MPI_COMM_WORLD);
    execlp("glmark2", "glmark2", "-b", bench_arg, (char*)NULL);
    MPI_Finalize();
    return 0;
}

