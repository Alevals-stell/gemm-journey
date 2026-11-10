#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <vector>
#include <random>
using namespace std;

void ijk(const float*A,const float*B,float*C,int n){
    for(int i=0;i<n;i++) for(int j=0;j<n;j++){
        float s=0; for(int k=0;k<n;k++) s+=A[i*n+k]*B[k*n+j];
        C[i*n+j]=s; }
}
void ikj(const float*A,const float*B,float*C,int n){
    for(int i=0;i<n;i++) for(int k=0;k<n;k++){
        float a=A[i*n+k]; for(int j=0;j<n;j++) C[i*n+j]+=a*B[k*n+j]; }
}

void init_random(float* X, int n){
    std::mt19937 gen(42);                                  // 引擎：产生随机比特
    std::uniform_real_distribution<float> dist(-1.f, 1.f); // 分布：把比特映射到 [-1,1] 
    for(int i = 0; i< n; ++i){
        X[i] = dist(gen);
    }

}

void matmul_naive_ijk(float* A, float* B, float* C, int M, int N, int K){
    for(int i=0; i < M; ++i){
        for(int j=0; j < N; ++j){
            float r = 0;
            for(int k=0; k < K; ++k){
                r += A[i*K+k] * B[k*N+j];
            }
            C[i*N+j] = r;
        }
    }
}

void matmul_naive_ikj(float* A, float* B, float* C, int M, int N, int K){
    for(int i=0; i < M; ++i){
        for(int k=0; k < K; ++k){
            float r = A[i*K+k];
            for(int j=0; j < N; ++j){
                C[i*N+j] += r * B[k*N+j];
            }
        }
    }
}

float check(const float* X, const float* Y, int n) {
    float max_err = 0.f;
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(X[i] - Y[i]);
        float err  = diff / std::max(std::fabs(Y[i]), 1.f);  // 分母至少为 1：大数用相对误差，小数退化为绝对误差
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// 子块版 ikj：计算 C(m×n) += A(m×k) * B(k×n)
// lda/ldb/ldc 是三个"大矩阵"每行实际的元素个数
void matmul_ikj_block(int m, int n, int k,
                      const float* A, int lda,
                      const float* B, int ldb,
                      float* C, int ldc) {
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            float a = A[i * lda + p];
            for (int j = 0; j < n; j++)
                C[i * ldc + j] += a * B[p * ldb + j];
        }
}

void matmul_tiled(int N, const float* A, const float* B, float* C, int BS) {
    // C 需要先清零
    for (int ii = 0; ii < N; ii += BS)
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS) {
                int m = std::min(BS, N - ii);
                int k = std::min(BS, N - kk);
                int n = std::min(BS, N - jj);
                matmul_ikj_block(m, n, k,
                                 A + ii * N + kk, N,
                                 B + kk * N + jj, N,
                                 C + ii * N + jj, N);
            }
}

int main(){
    
    int M = 1024, N = 1024, K = 1024, BS = 128;
    vector<float> A(M*K), B(K*N), C1(M*N), C2(M*N);
    init_random(A.data(), A.size());
    init_random(B.data(), B.size());

    auto bench = [&](auto f, vector<float>& C) {
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            fill(C.begin(), C.end(), 0.f);              // 每次清零，保证 ikj 的 += 正确
            auto t0 = chrono::steady_clock::now();
            f(A.data(), B.data(), C.data(), M, N, K);
            auto t1 = chrono::steady_clock::now();
            best = min(best, chrono::duration<double, milli>(t1 - t0).count());
        }
        return best;
    };

    auto bench1 = [&](auto f, vector<float>& C) {
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            fill(C.begin(), C.end(), 0.f);              // 每次清零，保证 ikj 的 += 正确
            auto t0 = chrono::steady_clock::now();
            f(N, A.data(), B.data(), C.data(), BS);
            auto t1 = chrono::steady_clock::now();
            best = min(best, chrono::duration<double, milli>(t1 - t0).count());
        }
        return best;
    };

    double t_ijk = 0.f;//bench(matmul_naive_ijk, C1);
    double t_ikj = bench(matmul_naive_ikj, C2);
    double tiled_t_ikj = bench1(matmul_tiled, C1);
    float  err   = check(C1.data(), C2.data(), M * N);

    double gflops_tiled_ikj = 2.0 * M * N * K / (tiled_t_ikj / 1000) / 1e9;
    double gflops_ikj = 2.0 * M * N * K / (t_ikj / 1000 )/ 1e9;

    printf("tiled_t_ikj: %.3f ms\nikj: %.3f ms\n加速比: %.2fx\n", gflops_tiled_ikj, gflops_ikj, t_ijk / t_ikj);
    printf("最大误差: %g\n", err);
   
    return 0;

}
