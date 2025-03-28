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

    //平凡算法
    for (int iter = 0; iter < ITERATIONS; iter++) {
        sum = 0;
        QueryPerformanceCounter((LARGE_INTEGER*)&head2);
        for (int i = 0; i < N; i++)
            sum += a[i];
        QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
        sum_total_time += (tail2 - head2) * 1000.0 / freq;
    }

    double sum_average_time = sum_total_time / ITERATIONS;
    cout << "Average time (Chain Sum): " << sum_average_time << "ms" << endl;

    // 释放动态分配的内存
    delete[] a;

    return 0;
}
