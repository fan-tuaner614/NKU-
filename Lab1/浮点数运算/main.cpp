#include <iostream>
using namespace std;

const int ARRAY_SIZE = 8;

// 串行累加函数
double serialSum(double numbers[], int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += numbers[i];
    }
    return sum;
}

// 并行累加函数
double parallelSum(double numbers[], int size) {
    if (size == 0) return 0.0;
    if (size == 1) return numbers[0];
    double partialSums[ARRAY_SIZE / 2 + 1];
    int partialIndex = 0;
    for (int i = 0; i < size; i += 2) {
        if (i + 1 < size) {
            partialSums[partialIndex++] = numbers[i] + numbers[i + 1];
        }
        else {
            partialSums[partialIndex++] = numbers[i];
        }
    }
    return serialSum(partialSums, partialIndex);
}


int main() {
    double numbers[ARRAY_SIZE] = { 1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16 };

    // 计算串行累加结果
    double serialResult = serialSum(numbers, ARRAY_SIZE);

    // 计算并行累加结果
    double parallelResult = parallelSum(numbers, ARRAY_SIZE);

    cout << "Serial sum: " << serialResult << endl;
    cout << "Parallel sum: " << parallelResult << endl;

    // 检查结果是否相同
    if (serialResult == parallelResult) {
        cout << "Results are the same." << endl;
    }
    else {
        cout << "Results are different." << endl;
    }

    return 0;
}
