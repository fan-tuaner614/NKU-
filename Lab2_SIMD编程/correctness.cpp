#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main

// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    cout << "MD5 Hash Test:" << endl;
    bit32 state1[4];
    MD5Hash("", state1);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state1[i1];
    }
    cout << endl;

    bit32 state2[4];
    MD5Hash("hello world", state2);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state2[i1];
    }
    cout << endl;

    bit32 state3[4];
    MD5Hash("!@#$%^&*()_+=-0987654321", state3);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state3[i1];
    }
    cout << endl;

    bit32 state4[4];
    MD5Hash("this is a long string for testing hash function. it should produce a unique hash value.", state4);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state4[i1];
    }
    cout << endl;

    cout << "MD5 Hash Test 2SIMD:" << endl;
    const int BATCH_SIZE0 = 2;
    alignas(16) uint32_t states0[BATCH_SIZE0][4];
    string pw0[BATCH_SIZE0] = {
        "!@#$%^&*()_+=-0987654321",
        "this is a long string for testing hash function. it should produce a unique hash value."};
    MD5Hash_2SIMD(pw0, states0, BATCH_SIZE0);

    for (int i = 0; i < BATCH_SIZE0; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            cout << std::setw(8) << std::setfill('0') << hex << states0[i][j];
        }
        cout << endl;
    }
    cout << endl;

    cout << "MD5 Hash Test SIMD:" << endl;
    const int BATCH_SIZE1 = 4;
    alignas(16) u_int32_t states1[BATCH_SIZE1][4];
    string pw1[BATCH_SIZE1] = {
        "",
        "hello world",
        "!@#$%^&*()_+=-0987654321",
        "this is a long string for testing hash function. it should produce a unique hash value."};
    MD5Hash_SIMD(pw1, states1, BATCH_SIZE1);

    for (int i = 0; i < BATCH_SIZE1; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            cout << std::setw(8) << std::setfill('0') << hex << states1[i][j];
        }
        cout << endl;
    }
    cout << endl;

    cout << "MD5 Hash Test no NEON SIMD:" << endl;
    const int BATCH_SIZE3 = 4;
    bit32 states3[BATCH_SIZE3][4];
    string pw3[BATCH_SIZE3] = {
        "",
        "hello world",
        "!@#$%^&*()_+=-0987654321",
        "this is a long string for testing hash function. it should produce a unique hash value."};
    MD5Hash_SIMD_noNEON(pw3, states3);
    for (int i = 0; i < BATCH_SIZE3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            cout << std::setw(8) << std::setfill('0') << hex << states3[i][j];
        }
        cout << endl;
    }
    cout << endl;

    cout << "MD5 Hash Test 8SIMD:" << endl;
    cout << endl;
    const int BATCH_SIZE2 = 8;
    alignas(16) uint32_t states2[BATCH_SIZE2][4];
    string pw2[BATCH_SIZE2] = {
        "",
        "hello world",
        "!@#$%^&*()_+=-0987654321",
        "this is a long string for testing hash function. it should produce a unique hash value.",
        "",
        "hello world",
        "!@#$%^&*()_+=-0987654321",
        "this is a long string for testing hash function. it should produce a unique hash value."};
    MD5Hash_8SIMD(pw2, states2, BATCH_SIZE2);

    for (int i = 0; i < BATCH_SIZE2; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            cout << std::setw(8) << std::setfill('0') << hex << states2[i][j];
        }
        cout << endl;
    }
    cout << endl;
}