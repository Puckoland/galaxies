#define BLOCK_DIM 5
#define BLOCK_SIZE BLOCK_DIM * BLOCK_DIM

__global__ void solve(sGalaxy A, sGalaxy B, float* distances, int n, int per_thread) {
    int base = BLOCK_SIZE * blockIdx.x;
    int bid = threadIdx.y * blockDim.x + threadIdx.x;
    int i = base + bid;

    __shared__ float D[BLOCK_SIZE];
    D[bid] = 0.0f;

    if (i >= n) return;
    float Ax = A.x[i];
    float Ay = A.y[i];
    float Az = A.z[i];
    float Bx = B.x[i];
    float By = B.y[i];
    float Bz = B.z[i];

    for (int t = blockIdx.x; t < per_thread; t++) {
        int index = t * BLOCK_SIZE + bid;

        __shared__ float As[BLOCK_SIZE * 3];
        __shared__ float Bs[BLOCK_SIZE * 3];

        // LOAD TO SHARED
        if (index < n) {
            As[bid * 3] = A.x[index];
            As[bid * 3 + 1] = A.y[index];
            As[bid * 3 + 2] = A.z[index];

            Bs[bid * 3] = B.x[index];
            Bs[bid * 3 + 1] = B.y[index];
            Bs[bid * 3 + 2] = B.z[index];
        }
        __syncthreads();

        // COMPUTE
        float tmp = 0.0f;
        int ooo = (t == blockIdx.x) ? bid + 1 : 0;
        for (int k = ooo; k < BLOCK_SIZE && base + k < n && t * BLOCK_SIZE + k < n; k++) {
            float da = sqrt(
                    (Ax - As[k * 3]) * (Ax - As[k * 3]) +
                    (Ay - As[k * 3 + 1]) * (Ay - As[k * 3 + 1]) +
                    (Az - As[k * 3 + 2]) * (Az - As[k * 3 + 2])
            );
            float db = sqrt(
                    (Bx - Bs[k * 3]) * (Bx - Bs[k * 3]) +
                    (By - Bs[k * 3 + 1]) * (By - Bs[k * 3 + 1]) +
                    (Bz - Bs[k * 3 + 2]) * (Bz - Bs[k * 3 + 2])
            );
            tmp += (da - db) * (da - db);
        }
        D[bid] += tmp;
    }
    __syncthreads();

    // REDUCE
    if (bid == 0) {
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
    int per_thread = roundUp(n, BLOCK_SIZE);
    int a = roundUp(n, BLOCK_SIZE);
    int b = 1;
    dim3 dimGrid (a, b);
    dim3 dimBlock (BLOCK_DIM, BLOCK_DIM);
    float* distances;
    size_t size = sizeof(*distances);
    cudaMalloc(&distances, size);
    cudaMemset(distances, 0, size);
    float* dist = (float *) malloc(size);
    if (dist == NULL) {
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    solve<<<dimGrid, dimBlock>>>(A, B, distances, n, per_thread);
    cudaMemcpy(dist, distances, size, cudaMemcpyDeviceToHost);

    float result = sqrt(1/((float)n*((float)n-1)) * *dist);
    free(dist);
    cudaFree(distances);
    return result;
}
