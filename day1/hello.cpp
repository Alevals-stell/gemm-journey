#include <cstdio>
#include <chrono>
#include <vector>

int main() {
    const int n = 100000000;
    std::vector<float> v(n, 1.0f);
    auto t0 = std::chrono::high_resolution_clock::now();
    double sum = 0;
    for (int i = 0; i < n; i++) sum += v[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("sum=%.0f, time=%.2f ms\n", sum, ms);
}
