#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        printf("CUDA error: %s at %s:%d\n",                     \
               cudaGetErrorString(err), __FILE__, __LINE__);    \
        exit(1);                                                \
    }                                                           \
} while (0)

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    // 补全：算全局编号，判断越界，做加法
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){ c[i] = a[i] + b[i]; }
}

int main() {
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 2.0f;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            if (errors < 5)
                printf("Mismatch at %d: gpu=%f, cpu=%f\n", i, h_c[i], h_a[i] + h_b[i]);
            errors++;
        }
    }
    printf(errors == 0 ? "PASS\n" : "FAIL (%d errors)\n", errors);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}

// 1. host 上分配 h_a, h_b, h_c，h_a、h_b 填充数据
    // 2. cudaMalloc 分配 d_a, d_b, d_c
    // 3. cudaMemcpy：h_a→d_a，h_b→d_b（方向 cudaMemcpyHostToDevice）
    // 4. 计算需要的 block 数，启动 kernel
    // 5. cudaMemcpy：d_c→h_c（方向 cudaMemcpyDeviceToHost）
    // 6. 在 CPU 上逐元素检查 h_c[i] == h_a[i] + h_b[i]，打印 PASS 或 FAIL
    // 7. cudaFree 三个 d_ 指针，释放 host 内存