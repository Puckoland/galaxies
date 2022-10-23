#define BLOCK_SIZE 256

void __global__ solve(sGalaxy A, sGalaxy B, float* distances, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int i = x % n;
    int j = i + 1 + x / n;

    __shared__ float D[BLOCK_SIZE];
    D[i] = 0.0f;

    if (j < n) {
        D[i] = sqrt((A.x[i] - A.x[j]) * (A.x[i] - A.x[j])
                    + (A.y[i] - A.y[j]) * (A.y[i] - A.y[j])
                    + (A.z[i] - A.z[j]) * (A.z[i] - A.z[j]))
               - sqrt((B.x[i] - B.x[j]) * (B.x[i] - B.x[j])
                      + (B.y[i] - B.y[j]) * (B.y[i] - B.y[j])
                      + (B.z[i] - B.z[j]) * (B.z[i] - B.z[j]));
        printf("%d %d: %f\n", i, j, D[i]);
    }

    __syncthreads();
    if (i == 0) {
        for (int o = 0; o < n; o++) {
            distances[blockIdx.x] += D[i] * D[i];
        }
    }
}

int roundUp(int value, int div) {
    if (value % div == 0) {
        return value / div;
    }
    return value / div + 1;
}

float solveGPU(sGalaxy A, sGalaxy B, int n) {
    int blocks = roundUp(n, BLOCK_SIZE);
    float* distances;
    size_t size = blocks * sizeof(*distances);
    cudaMalloc(&distances, size);
    float* dist = (float *) malloc(size);
    if (dist == NULL) {
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    solve<<<1, BLOCK_SIZE>>>(A, B, distances, n);
    cudaMemcpy(dist, distances, size, cudaMemcpyDeviceToHost);

    float result = 0.0f;
    printf("TTTTT\n");
    for (int i = 0; i < blocks; i++) {
        printf("%f\n", dist[i]);
        result += dist[i];
    }
    return sqrt(result / n / (n-1));
}
