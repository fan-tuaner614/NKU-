#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <arm_neon.h>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main_simd.cpp train.cpp guessing.cpp md5_simd.cpp -o main
// g++ main_simd.cpp train.cpp guessing.cpp md5_simd.cpp -o main -O1
// g++ main_simd.cpp train.cpp guessing.cpp md5_simd.cpp -o main -O2
#include <iostream>
#include <vector>
#include <chrono>
#include <arm_neon.h>
#include <cassert>
#include <cstring>

// NEON优化的MD5函数声明
void MD5Hash_SIMD_noNEON(string inputs[4], bit32 states[4][4]);

int main()
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长

    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;

    // std::ofstream a("./files/results.txt");
    // cout << q.guesses.size() << endl;
    // cout << q.priority.size() << endl;
    while (!q.priority.empty())
    {
        // cout << "q.priority.size(): " << q.priority.size() << ", q.guesses.size(): " << q.guesses.size() << endl;
        q.PopNext();
        q.total_guesses = q.guesses.size();
        // cout << "q.total_guesses: " << q.total_guesses << ", curr_num: " << curr_num << endl;
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n = 10000000;
            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            const int BATCH_SIZE = 4;
            string pw[BATCH_SIZE];
            const int aligned_size = (q.guesses.size() / BATCH_SIZE) * BATCH_SIZE;
            // cout << q.guesses.size() << endl;
            alignas(16) uint32_t states[BATCH_SIZE][4];
            auto start_hash = system_clock::now();

            for (int i = 0; i < aligned_size; i += BATCH_SIZE)
            {
                pw[0] = q.guesses[i];
                pw[1] = q.guesses[i + 1];
                pw[2] = q.guesses[i + 2];
                pw[3] = q.guesses[i + 3];
                MD5Hash_SIMD(pw, states);
            }

            // 处理剩余元素
            for (int i = aligned_size; i < q.guesses.size(); ++i)
            {
                // cout << q.guesses.size() - aligned_size << endl;
                bit32 state[4];
                MD5Hash(q.guesses[i], state); // 原始单次处理
            }

            //  在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
            // cout << "After clear: history = " << history << ", curr_num = " << curr_num << endl;
        }
    }
}