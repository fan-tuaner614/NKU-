#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const int N = 1000; // ����������Ĵ�С,���޸�
const int ITERATIONS = 1000; // ִ�д���,���޸�
const int UNROLL_FACTOR = 4; // ѭ��չ������,���޸�

int main() {
    std::cout << "N: "<<N << endl;
    // ��̬������������
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
    double total_time = 0.0;

    // ƽ���㷨
    for (int iter = 0; iter < ITERATIONS; iter++) {
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
    std::cout << "Average time (Col): " << average_time << "ms" << endl;

    long long head1, tail1;
    total_time = 0.0;
    for (int i = 0; i < N; i++) {
        result[i] = 0.0;
    }
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    // ƽ���㷨��ѭ��չ���Ż���
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
    //cache�Ż��㷨
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

    // ����ƽ������ʱ��
    average_time = total_time / ITERATIONS;
    std::cout << "Average time (Row with unroll): " << average_time << "ms" << endl;

    // �ͷŶ�̬������ڴ�
    for (int i = 0; i < N; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] vector;
    delete[] result;

    return 0;
}
