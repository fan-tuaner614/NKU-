#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const int N = 1000; // 矩阵和向量的大小
const int ITERATIONS = 1000; // 执行次数

int main() {
    // 动态分配矩阵和向量
    cout<<"N:"<<N<<endl;
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
    double total_time = 0.0; // 总运行时间

    // 平凡算法
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // 开始计时
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
    cout << "Average time (Col): " << average_time << "ms" << endl;
     // 释放动态分配的内存
    for (int i = 0; i < N; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] vector;
    delete[] result;

    return 0;
}
