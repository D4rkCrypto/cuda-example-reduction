#include "util.cuh"

__global__ void reduce2(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void reduce2() {
    int *h_i_data, *h_o_data;
    int *d_i_data, *d_o_data;
    int n = 1 << 22;
    int threads = 128;   // initial block size
    size_t nBlocks = n / threads + (n % threads == 0 ? 0 : 1);
    size_t nBytes = n * sizeof(int);
    size_t smemSize = threads * sizeof(int);

    // allocate host memory
    h_i_data = (int*)malloc(nBytes);
    h_o_data = (int*)malloc(nBlocks * sizeof(int));

    // allocate device memory
    cudaMalloc((void**)&d_i_data, nBytes);
    cudaMalloc((void**)&d_o_data, nBlocks * sizeof(int));

    // initialize host memory
    for (int i=0; i < n; i++)
        h_i_data[i] = my_rand_int(-32, 31);

    // copy host memory to device
    cudaMemcpy(d_i_data, h_i_data, nBytes, cudaMemcpyHostToDevice);

    dim3 dimGrid(nBlocks, 1, 1);
    dim3 dimBlock(threads, 1 , 1);

    // execute the kernel
    CUDATimer timer = CUDATimer("reduce2");
    for (int i=0; i < N_TESTS; i++) {
        timer.start();
        reduce2<<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data);
        timer.stop();
    }

    // copy result from device to host
    cudaMemcpy(h_o_data, d_o_data, nBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    int i_sum = std::reduce(h_i_data, h_i_data + n, 0, std::plus<>());
    int o_sum = std::reduce(h_o_data, h_o_data + nBlocks, 0, std::plus<>());
    if (i_sum != o_sum)
        std::cout << "Incorrect." << std::endl;

    // cleanup memory
    free(h_i_data); free(h_o_data);
    cudaFree(d_i_data); cudaFree(d_o_data);
}
