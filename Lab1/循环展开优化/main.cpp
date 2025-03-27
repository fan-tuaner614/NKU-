#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const int N = 1000; // 矩阵和向量的大小,可修改
const int ITERATIONS = 1000; // 执行次数,可修改
const int UNROLL_FACTOR = 4; // 循环展开因子,可修改

int main() {
    std::cout << "N: "<<N << endl;
    // 动态分配矩阵和向量
    double** matrix = new double* [N];
    for (int i = 0; i < N; i++) {
        matrix[i] = new double[N];
    }
    double* vector = new double[N];
    double* result = new double[N](); // 结果数组，初始化为 0

    // 初始化矩阵和向量
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = i + j;
        }
        vector[i] = i;
    }

    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    double total_time = 0.0;

    // 平凡算法
    for (int iter = 0; iter < ITERATIONS; iter++) {
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        // 逐列访问矩阵并与向量进行内积计算
        for (int i = 0; i < N; i++) {
            result[i] = 0.0;
            for (int j = 0; j < N; j++) {
                result[i] += matrix[j][i] * vector[j];
            }
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        total_time += (tail - head) * 1000.0 / freq;
    }

    // 计算平均运行时间
    double average_time = total_time / ITERATIONS;
    std::cout << "Average time (Col): " << average_time << "ms" << endl;

    long long head1, tail1;
    total_time = 0.0;
    for (int i = 0; i < N; i++) {
        result[i] = 0.0;
    }
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    // 平凡算法（循环展开优化）
    for (int iter = 0; iter < ITERATIONS; iter++) {
        QueryPerformanceCounter((LARGE_INTEGER*)&head1);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j += UNROLL_FACTOR) {
                result[i] += matrix[j][i] * vector[j];
                result[i] += matrix[j+1][i] * vector[j+1];
                result[i] += matrix[j+2][i] * vector[j+2];
                result[i] += matrix[j+3][i] * vector[j+3];
            }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
        total_time += (tail1 - head1) * 1000.0 / freq;
    }

    std::cout << "Average time (Col with unroll): " << total_time / ITERATIONS << "ms" << endl;

    total_time = 0;
    long long head2, tail2;
    for (int i = 0; i < N; i++) {
        result[i] = 0.0;
    }
    //cache优化算法
    for (int iter = 0; iter < ITERATIONS; iter++) {
        QueryPerformanceCounter((LARGE_INTEGER*)&head2);
        for (int j = 0; j < N; j += UNROLL_FACTOR)
            for (int i = 0; i < N; i++) {
                result[i] += matrix[j][i] * vector[j];
                if (j + 1 < N) result[i] += matrix[j + 1][i] * vector[j + 1];
                if (j + 2 < N) result[i] += matrix[j + 2][i] * vector[j + 2];
                if (j + 3 < N) result[i] += matrix[j + 3][i] * vector[j + 3];
            }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
        total_time += (tail2 - head2) * 1000.0 / freq;
    }

    // 计算平均运行时间
    average_time = total_time / ITERATIONS;
    std::cout << "Average time (Row with unroll): " << average_time << "ms" << endl;

    // 释放动态分配的内存
    for (int i = 0; i < N; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] vector;
    delete[] result;

    return 0;
}
