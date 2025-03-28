#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const int N = 100000; // 数组的大小
const int ITERATIONS = 1000; // 执行次数


int main() {
    int* a = new int[N];
    // 初始化
    for (int i = 0; i < N; i++) {
        a[i] = i;
    }
    long long sum = 0;
    cout<<N<<endl;
    long long freq ,head2, tail2;
    double sum_total_time = 0.0;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    // 多链路算法
    long long head3, tail3;
    double multi_chain_total_time = 0.0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        long long sum1 = 0, sum2 = 0;
        QueryPerformanceCounter((LARGE_INTEGER*)&head3);
        for (int i = 0; i < N; i += 2) {
            sum1 += a[i];
            sum2 += a[i + 1];
        }
        long long multi_chain_sum = sum1 + sum2;
        QueryPerformanceCounter((LARGE_INTEGER*)&tail3);
        multi_chain_total_time += (tail3 - head3) * 1000.0 / freq;
    }
    double multi_chain_average_time = multi_chain_total_time / ITERATIONS;
    cout << "Average time (Multi - Chain Sum): " << multi_chain_average_time << "ms" << endl;

    // 释放动态分配的内存
    delete[] a;

    return 0;
}
