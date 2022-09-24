#include "util.cuh"

template <unsigned int blockSize>
__device__ void warpReduce7(volatile int * sdata, int tid){
    if (blockSize >= 64) sdata[tid] += sdata[tid +32];
    if (blockSize >= 32) sdata[tid] += sdata[tid +16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +1];
}

template <unsigned int blockSize>
__global__ void reduce7(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
            __syncthreads();
        }
    }

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
            __syncthreads();
        }
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
            __syncthreads();
        }
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
            __syncthreads();
        }
    }

    if (tid < 32) warpReduce7<blockSize>(sdata, tid);

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void reduce7() {
    int *h_i_data, *h_o_data;
    int *d_i_data, *d_o_data;
    int n = 1 << 22;
    int threads = 128;   // initial block size
    size_t nBlocks = n / threads / 2 + (n % threads == 0 ? 0 : 1);
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
    CUDATimer timer = CUDATimer("reduce7");
    for (int i=0; i < N_TESTS; i++) {
        timer.start();
        switch (threads)
        {
            case 1024:
                reduce7<1024><<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data, n);
                break;
            case 512:
                reduce7<512><<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data, n);
                break;
            case 256:
                reduce7<256><<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data, n);
                break;
            case 128:
                reduce7<128><<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data, n);
                break;
            case 64:
                reduce7<64><<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data, n);
                break;
            case 32:
                reduce7<32><<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data, n);
                break;
            case 16:
                reduce7<16><<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data, n);
                break;
            case 8:
                reduce7<8><<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data, n);
                break;
            case 4:
                reduce7<4><<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data, n);
                break;
            case 2:
                reduce7<2><<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data, n);
                break;
            case 1:
                reduce7<1><<<dimGrid, dimBlock, smemSize>>>(d_i_data, d_o_data, n);
                break;
            default:
                break;
        }
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
