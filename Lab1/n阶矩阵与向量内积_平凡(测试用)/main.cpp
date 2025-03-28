#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const int N = 1000; // ����������Ĵ�С
const int ITERATIONS = 1000; // ִ�д���

int main() {
    // ��̬������������
    cout<<"N:"<<N<<endl;
    double** matrix = new double* [N];
    for (int i = 0; i < N; i++) {
        matrix[i] = new double[N];
    }
    double* vector = new double[N];
    double* result = new double[N](); // ������飬��ʼ��Ϊ 0

    // ��ʼ�����������
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = i + j;
        }
        vector[i] = i;
    }

    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    double total_time = 0.0; // ������ʱ��

    // ƽ���㷨
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // ��ʼ��ʱ
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        // ���з��ʾ��������������ڻ�����
        for (int i = 0; i < N; i++) {
            result[i] = 0.0;
            for (int j = 0; j < N; j++) {
                result[i] += matrix[j][i] * vector[j];
            }
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        total_time += (tail - head) * 1000.0 / freq;
    }

    // ����ƽ������ʱ��
    double average_time = total_time / ITERATIONS;
    cout << "Average time (Col): " << average_time << "ms" << endl;
     // �ͷŶ�̬������ڴ�
    for (int i = 0; i < N; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] vector;
    delete[] result;

    return 0;
}
