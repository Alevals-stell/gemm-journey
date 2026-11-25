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

struct Item { float a, b, c; };   // AoS：一个元素的三个值挨在一起，共 12 字节

// ---------- kernels ----------
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vec_add_aos(Item* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i].c = d[i].a + d[i].b;
}

// 每个线程处理 4 个 float，n4 = n / 4
__global__ void vec_add_f4(const float4* a, const float4* b, float4* c, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

// 辅助 kernel：数组 <-> AoS 的转换（不参与计时）
__global__ void pack_aos(const float* a, const float* b, Item* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { d[i].a = a[i]; d[i].b = b[i]; d[i].c = 0.0f; }
}
__global__ void unpack_c(const Item* d, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = d[i].c;
}

// ---------- 工具函数 ----------
void result_check(const float* h_a, const float* h_b, const float* h_c, int n) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            if (errors < 5)
                printf("  Mismatch at %d: gpu=%f, cpu=%f\n", i, h_c[i], h_a[i] + h_b[i]);
            errors++;
        }
    }
    printf(errors == 0 ? "  PASS\n" : "  FAIL (%d errors)\n", errors);
}

// 通用计时：launch 是一个"启动一次操作"的函数，返回每次平均耗时（毫秒）
template <typename F>
float time_avg_ms(F launch, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 5; i++) launch();               // 预热
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

void report(const char* name, float avg_ms, double bytes_moved) {
    double bw = bytes_moved / (avg_ms / 1000.0) / 1e9;   // GB/s
    printf("  %-16s %8.3f ms  %8.2f GB/s\n", name, avg_ms, bw);
}

// ---------- main ----------
int main(int argc, char** argv) {
    int n = 100000000;                  // 必须是 4 的倍数（float4 版本要求）
    int threads = 256;
    if (argc > 1) threads = atoi(argv[1]);
    if (n % 4 != 0) { printf("n must be a multiple of 4\n"); return 1; }

    int iters = 20;
    size_t bytes = (size_t)n * sizeof(float);
    int n4 = n / 4;
    int blocks  = (n  + threads - 1) / threads;
    int blocks4 = (n4 + threads - 1) / threads;
    printf("n=%d, threads=%d\n", n, threads);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { h_a[i] = i * 0.5f; h_b[i] = i * 2.0f; }

    float *d_a, *d_b, *d_c;
    Item *d_item;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_item, (size_t)n * sizeof(Item)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 每次校验前先把 d_c 清零，这样"没算"和"算错"能区分开
    auto verify = [&]() {
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
        result_check(h_a, h_b, h_c, n);
    };

    printf("[1] SoA scalar (原版)\n");
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    verify();
    float t1 = time_avg_ms([&]{ vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n); }, iters);
    report("SoA scalar", t1, 12.0 * n);

    printf("[2] AoS\n");
    pack_aos<<<blocks, threads>>>(d_a, d_b, d_item, n);
    vec_add_aos<<<blocks, threads>>>(d_item, n);
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    unpack_c<<<blocks, threads>>>(d_item, d_c, n);
    verify();
    float t2 = time_avg_ms([&]{ vec_add_aos<<<blocks, threads>>>(d_item, n); }, iters);
    report("AoS", t2, 12.0 * n);

    printf("[3] float4\n");
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    vec_add_f4<<<blocks4, threads>>>((const float4*)d_a, (const float4*)d_b, (float4*)d_c, n4);
    verify();
    float t3 = time_avg_ms([&]{
        vec_add_f4<<<blocks4, threads>>>((const float4*)d_a, (const float4*)d_b, (float4*)d_c, n4);
    }, iters);
    report("float4", t3, 12.0 * n);

    printf("[4] D2D memcpy (实际带宽参照，读+写各一份)\n");
    float t4 = time_avg_ms([&]{
        CUDA_CHECK(cudaMemcpy(d_c, d_a, bytes, cudaMemcpyDeviceToDevice));
    }, iters);
    report("D2D memcpy", t4, 2.0 * bytes);

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c)); CUDA_CHECK(cudaFree(d_item));
    free(h_a); free(h_b); free(h_c);
    return 0;
}