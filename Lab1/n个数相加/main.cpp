#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const int N = 10000; // 数组的大小
const int ITERATIONS = 1000; // 执行次数

// 递归加法函数
void recursion(int* a, int n) {
    if (n == 1) return;
    for (int i = 0; i < n / 2; i++)
        a[i] += a[n - i - 1];
    recursion(a, n / 2);
}

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

    // 递归算法
    long long head4, tail4;
    double recursion_total_time = 0.0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        int* temp_a = new int[N];
        for (int i = 0; i < N; i++) {
            temp_a[i] = a[i];
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&head4);
        recursion(temp_a, N);
        long long recursion_sum = temp_a[0];
        QueryPerformanceCounter((LARGE_INTEGER*)&tail4);
        recursion_total_time += (tail4 - head4) * 1000.0 / freq;
        delete[] temp_a;
    }
    double recursion_average_time = recursion_total_time / ITERATIONS;
    cout << "Average time (Recursion Sum): " << recursion_average_time << "ms" << endl;

    // 二重循环算法
    long long head5, tail5;
    double double_loop_total_time = 0.0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        int* temp_a = new int[N];
        for (int i = 0; i < N; i++) {
            temp_a[i] = a[i];
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&head5);
        for (int m = N; m > 1; m /= 2)
            for (int i = 0; i < m / 2; i++)
                temp_a[i] = temp_a[i * 2] + temp_a[i * 2 + 1];
        long long double_loop_sum = temp_a[0];
        QueryPerformanceCounter((LARGE_INTEGER*)&tail5);
        double_loop_total_time += (tail5 - head5) * 1000.0 / freq;
        delete[] temp_a;
    }
    double double_loop_average_time = double_loop_total_time / ITERATIONS;
    cout << "Average time (Double - Loop Sum): " << double_loop_average_time << "ms" << endl;

    // 释放动态分配的内存
    delete[] a;

    return 0;
}
