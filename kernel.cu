#define BLOCK_DIM 22
#define BLOCK_SIZE 484

__global__ void solve(sGalaxy A, sGalaxy B, float* distances, int n) {
//    printf("GRID: %d %d\n", gridDim.x, gridDim.y);
    int bid = threadIdx.y * blockDim.x + threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int ii = blockDim.y * blockIdx.y + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float D[BLOCK_SIZE];
    D[bid] = 0.0f;

    __shared__ float As[BLOCK_DIM][3][2];
    __shared__ float Bs[BLOCK_DIM][3][2];

    if (threadIdx.y == 0) {
        As[threadIdx.x][0][0] = A.x[i];
        As[threadIdx.x][1][0] = A.y[i];
        As[threadIdx.x][2][0] = A.z[i];
    } else if (threadIdx.y == 1) {
        As[threadIdx.x][0][1] = A.x[ii];
        As[threadIdx.x][1][1] = A.y[ii];
        As[threadIdx.x][2][1] = A.z[ii];
    } else if (threadIdx.y == 2) {
        Bs[threadIdx.x][0][0] = B.x[i];
        Bs[threadIdx.x][1][0] = B.y[i];
        Bs[threadIdx.x][2][0] = B.z[i];
    } else if (threadIdx.y == 3) {
        Bs[threadIdx.x][0][1] = B.x[ii];
        Bs[threadIdx.x][1][1] = B.y[ii];
        Bs[threadIdx.x][2][1] = B.z[ii];
    }
    __syncthreads();

    if (i < n && j < n && i < j) {
        float da = sqrt(
                (As[threadIdx.x][0][0]-As[threadIdx.y][0][1])*(As[threadIdx.x][0][0]-As[threadIdx.y][0][1])
                + (As[threadIdx.x][1][0]-As[threadIdx.y][1][1])*(As[threadIdx.x][1][0]-As[threadIdx.y][1][1])
                + (As[threadIdx.x][2][0]-As[threadIdx.y][2][1])*(As[threadIdx.x][2][0]-As[threadIdx.y][2][1])
                );
        float db = sqrt(
                (Bs[threadIdx.x][0][0]-Bs[threadIdx.y][0][1])*(Bs[threadIdx.x][0][0]-Bs[threadIdx.y][0][1])
                + (Bs[threadIdx.x][1][0]-Bs[threadIdx.y][1][1])*(Bs[threadIdx.x][1][0]-Bs[threadIdx.y][1][1])
                + (Bs[threadIdx.x][2][0]-Bs[threadIdx.y][2][1])*(Bs[threadIdx.x][2][0]-Bs[threadIdx.y][2][1])
        );
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
