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
        result[i] = 0;
    }

    long long head1, tail1,freq;
    double total_time = 0.0;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // cache优化算法
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // 开始计时
        QueryPerformanceCounter((LARGE_INTEGER*)&head1);

        // 改为逐行访问矩阵元素：一步外层循环计算不出任何一个内积，只是向每个内积累加一个乘法结果

        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                result[i] += matrix[j][i] * vector[j];

        QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
        total_time += (tail1 - head1) * 1000.0 / freq;
    }

    // 计算平均运行时间
    double average_time = total_time / ITERATIONS;
    cout << "Average time (Row): " << average_time << "ms" << endl;

    // 释放动态分配的内存
    for (int i = 0; i < N; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] vector;
    delete[] result;

    return 0;
}
