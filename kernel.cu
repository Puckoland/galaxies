#define BLOCK_SIZE 484

__global__ void solve(sGalaxy A, sGalaxy B, float* distances, int n) {
//    printf("GRID: %d %d\n", gridDim.x, gridDim.y);
    int bid = threadIdx.y * blockDim.x + threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float D[BLOCK_SIZE];
    D[bid] = 0.0f;

    if (i < n && j < n && i < j) {
        float da = sqrt((A.x[i]-A.x[j])*(A.x[i]-A.x[j])
                        + (A.y[i]-A.y[j])*(A.y[i]-A.y[j])
                        + (A.z[i]-A.z[j])*(A.z[i]-A.z[j]));
        float db = sqrt((B.x[i]-B.x[j])*(B.x[i]-B.x[j])
                        + (B.y[i]-B.y[j])*(B.y[i]-B.y[j])
                        + (B.z[i]-B.z[j])*(B.z[i]-B.z[j]));
        D[bid] = (da-db) * (da-db);
//        printf("(%d %d) D[%d] = %f\n", i, j, bid, D[bid]);
    }

    __syncthreads();
    if (bid == 0) {
        float sum = 0.0f;
        for (int o = 0; o < BLOCK_SIZE; o++) {
            sum += D[o];
        }
//        printf("SUM (%d %d): %f\n", blockIdx.x, blockIdx.y, sum);
        atomicAdd(distances, sum);
    }
}

int roundUp(int value, int div) {
    if (value % div == 0) {
        return value / div;
    }
    return value / div + 1;
}

float solveGPU(sGalaxy A, sGalaxy B, int n) {
    int a = roundUp(n, 22);
    dim3 dimGrid (a, a);
    dim3 dimBlock (22, 22);
    float* distances;
    size_t size = sizeof(*distances);
    cudaMalloc(&distances, size);
    cudaMemset(distances, 0, size);
    float* dist = (float *) malloc(size);
    if (dist == NULL) {
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    solve<<<dimGrid, dimBlock>>>(A, B, distances, n);
    cudaMemcpy(dist, distances, size, cudaMemcpyDeviceToHost);

    float result = sqrt(1/((float)n*((float)n-1)) * *dist);
    free(dist);
    cudaFree(distances);
    return result;
}
