#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

struct Item { float a, b, c; };

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


__global__ void vec_add_aos(Item* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i].c = d[i].a + d[i].b;
}

__global__ void vec_add4(const float4* a, const float4* b, float4* c, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

void result_check(const float* h_a, const float* h_b, const float* h_c, int n){
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            if (errors < 5)
                printf("Mismatch at %d: gpu=%f, cpu=%f\n", i, h_c[i], h_a[i] + h_b[i]);
            errors++;
        }
    }
    printf(errors == 0 ? "PASS\n" : "FAIL (%d errors)\n", errors);
}

int main(){
    int n = 100000000;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 2.0f;
    }
// 1. 建两个 event
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float H2D_time;
    CUDA_CHECK(cudaEventElapsedTime(&H2D_time, start, stop));
    printf("H2D_time: %.3f\n", H2D_time/1000.0);
    
    

    // 2. 预热：先跑几次不计时（原因见下面的提示）
    for (int i = 0; i < 5; i++) vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 3. 计时：补全 record 和同步，循环 20 次
    int iters = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    // 补全：等 stop 事件完成，再取时间
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float D2H_time;
    CUDA_CHECK(cudaEventElapsedTime(&D2H_time, start, stop));
    printf("D2H_time: %.3f\n", D2H_time/1000.0);
//检查结果是否有误  
    result_check(h_a, h_b, h_c, n);
    
    double arg_t = ms / iters / 1000.0;
    double moved = 3 * n * sizeof(float);
    double bandwidth =  moved / arg_t/ 1e9;
    double gflops = n / arg_t / 1e9;
    // 4. 计算：平均每次耗时、有效带宽 GB/s
    // 数据量 = 读 a、读 b、写 c = 3 * n * sizeof(float) 字节
    // 带宽(GB/s) = 数据量 / 秒 / 1e9
    printf("time: %.3f\ngflops: %.3fGFLOPS\nbandwidth: %.3fGB/s\n", arg_t, gflops, bandwidth);
    
    CUDA_CHECK(cudaEventDestroy(start)); 
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;

}