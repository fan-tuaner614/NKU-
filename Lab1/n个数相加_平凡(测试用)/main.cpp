#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const int N = 100000; // ����Ĵ�С
const int ITERATIONS = 1000; // ִ�д���

int main() {
    int* a = new int[N];
    // ��ʼ��
    for (int i = 0; i < N; i++) {
        a[i] = i;
    }
    long long sum = 0;
    cout<<N<<endl;
    long long freq ,head2, tail2;
    double sum_total_time = 0.0;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    //ƽ���㷨
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

    // �ͷŶ�̬������ڴ�
    delete[] a;

    return 0;
}
