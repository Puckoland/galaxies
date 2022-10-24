#define BLOCK_SIZE 256

__global__ void solve(sGalaxy A, sGalaxy B, float* distances, int n) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int i = x % n;
    int j = i + 1 + x / n;

    __shared__ float D[BLOCK_SIZE];
    D[threadIdx.x] = 0.0f;

    if (j < n) {
        float da = sqrt((A.x[i]-A.x[j])*(A.x[i]-A.x[j])
                        + (A.y[i]-A.y[j])*(A.y[i]-A.y[j])
                        + (A.z[i]-A.z[j])*(A.z[i]-A.z[j]));
        float db = sqrt((B.x[i]-B.x[j])*(B.x[i]-B.x[j])
                        + (B.y[i]-B.y[j])*(B.y[i]-B.y[j])
                        + (B.z[i]-B.z[j])*(B.z[i]-B.z[j]));
        D[threadIdx.x] = (da-db) * (da-db);
//        if (i == 0) printf("D[%d] (%d %d) = %f\n", bid, i, j, D[bid]);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int o = 0; o < BLOCK_SIZE; o++) {
            sum += D[o];
        }
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
    dim3 dimGrid = roundUp(n*n, BLOCK_SIZE);
    dim3 dimBlock = BLOCK_SIZE;
    float* distances;
    size_t size = sizeof(*distances);
    cudaMalloc(&distances, size);
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
