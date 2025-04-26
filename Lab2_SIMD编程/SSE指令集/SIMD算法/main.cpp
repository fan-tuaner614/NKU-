#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <cassert>
#include <iostream>
#include <iomanip>

using Byte = unsigned char;
using bit32 = uint32_t;

// MD5 标准初始值
const bit32 INIT_A = 0x67452301;
const bit32 INIT_B = 0xefcdab89;
const bit32 INIT_C = 0x98badcfe;
const bit32 INIT_D = 0x10325476;

// 移位常量
const int s11 = 7, s12 = 12, s13 = 17, s14 = 22;
const int s21 = 5, s22 = 9, s23 = 14, s24 = 20;
const int s31 = 4, s32 = 11, s33 = 16, s34 = 23;
const int s41 = 6, s42 = 10, s43 = 15, s44 = 21;

// MD5 轮函数中的常数（T表）
const bit32 T[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

// 静态缓冲区
alignas(16) static bit32 interleaved_pool4[16 * 1024 * 4]; // 支持最多 1024 块
alignas(16) static bit32 mask_pool4[1024 * 4];             // 支持最多 1024 块

// 消息填充函数
Byte* StringProcess(const std::string& input, int* paddedLen) {
    int origLen = input.length();
    int totalLen = ((origLen + 8) / 64 + 1) * 64;
    Byte* padded = new Byte[totalLen]();
    memcpy(padded, input.data(), origLen);
    padded[origLen] = 0x80;
    uint64_t bitLen = (uint64_t)origLen * 8;
    memcpy(padded + totalLen - 8, &bitLen, 8);
    *paddedLen = totalLen;
    return padded;
}

// SSE 轮函数宏定义（与标准 MD5 和 NEON 版本一致）
#define FF_SIMD(a, b, c, d, x, s, t) do { \
    __m128i temp = _mm_and_si128(b, c); \
    temp = _mm_or_si128(temp, _mm_andnot_si128(b, d)); \
    temp = _mm_add_epi32(temp, a); \
    temp = _mm_add_epi32(temp, x); \
    temp = _mm_add_epi32(temp, _mm_set1_epi32(t)); \
    a = _mm_add_epi32(b, _mm_or_si128(_mm_slli_epi32(temp, s), _mm_srli_epi32(temp, 32 - s))); \
} while(0)

#define GG_SIMD(a, b, c, d, x, s, t) do { \
    __m128i temp = _mm_and_si128(d, b); \
    temp = _mm_or_si128(temp, _mm_andnot_si128(d, c)); \
    temp = _mm_add_epi32(temp, a); \
    temp = _mm_add_epi32(temp, x); \
    temp = _mm_add_epi32(temp, _mm_set1_epi32(t)); \
    a = _mm_add_epi32(b, _mm_or_si128(_mm_slli_epi32(temp, s), _mm_srli_epi32(temp, 32 - s))); \
} while(0)

#define HH_SIMD(a, b, c, d, x, s, t) do { \
    __m128i temp = _mm_xor_si128(b, c); \
    temp = _mm_xor_si128(temp, d); \
    temp = _mm_add_epi32(temp, a); \
    temp = _mm_add_epi32(temp, x); \
    temp = _mm_add_epi32(temp, _mm_set1_epi32(t)); \
    a = _mm_add_epi32(b, _mm_or_si128(_mm_slli_epi32(temp, s), _mm_srli_epi32(temp, 32 - s))); \
} while(0)

#define II_SIMD(a, b, c, d, x, s, t) do { \
    __m128i temp = _mm_xor_si128(c, _mm_or_si128(b, _mm_andnot_si128(d, _mm_set1_epi32(-1)))); \
    temp = _mm_add_epi32(temp, a); \
    temp = _mm_add_epi32(temp, x); \
    temp = _mm_add_epi32(temp, _mm_set1_epi32(t)); \
    a = _mm_add_epi32(b, _mm_or_si128(_mm_slli_epi32(temp, s), _mm_srli_epi32(temp, 32 - s))); \
} while(0)

// 字节反转函数，使用 SSE2 兼容实现，模拟 vrev32q_u8
__m128i ByteSwap(__m128i vec) {
    alignas(16) bit32 temp[4];
    _mm_store_si128((__m128i*)temp, vec);
    for (int i = 0; i < 4; ++i) {
        bit32 val = temp[i];
        temp[i] = ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) |
                  ((val & 0xFF0000) >> 8) | ((val & 0xFF000000) >> 24);
    }
    return _mm_load_si128((__m128i*)temp);
}

// SSE 版本的 MD5 哈希函数
void MD5Hash_SSE(const std::string* inputs, bit32 states[][4], int batch_size) {
    // 验证 batch_size
    assert(batch_size == 4 && "MD5Hash_SSE requires batch_size=4 for SIMD optimization");

    // 处理输入字符串
    Byte** paddedMessages = new Byte*[batch_size];
    int* messageLength = new int[batch_size];
    int max_blocks = 0;
    for (int k = 0; k < batch_size; ++k) {
        paddedMessages[k] = StringProcess(inputs[k], &messageLength[k]);
        int n_blocks_k = messageLength[k] / 64;
        if (n_blocks_k > max_blocks) max_blocks = n_blocks_k;
    }

    // 初始化状态向量
    __m128i state_a = _mm_set1_epi32(INIT_A);
    __m128i state_b = _mm_set1_epi32(INIT_B);
    __m128i state_c = _mm_set1_epi32(INIT_C);
    __m128i state_d = _mm_set1_epi32(INIT_D);

    // 创建掩码数组
    bit32* mask_array = mask_pool4;
    for (int i = 0; i < max_blocks; i += 4) {
        for (int k = 0; k < batch_size; ++k) {
            mask_array[i * batch_size + k] = (i < messageLength[k] / 64) ? 0xFFFFFFFF : 0;
            mask_array[(i + 1) * batch_size + k] = ((i + 1) < messageLength[k] / 64) ? 0xFFFFFFFF : 0;
            mask_array[(i + 2) * batch_size + k] = ((i + 2) < messageLength[k] / 64) ? 0xFFFFFFFF : 0;
            mask_array[(i + 3) * batch_size + k] = ((i + 3) < messageLength[k] / 64) ? 0xFFFFFFFF : 0;
        }
    }

    // 使用标量操作交错消息块，精确模拟 NEON 的 vld1q_u8 和 vst1q_lane_u32
    bit32* interleaved = interleaved_pool4;
    for (int i = 0; i < max_blocks; ++i) {
        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < batch_size; ++k) {
                if (i < messageLength[k] / 64) {
                    // 标量加载 4 字节，确保小端字节序
                    bit32 word = 0;
                    Byte* src = &paddedMessages[k][i * 64 + j * 4];
                    word = (bit32)src[0] | ((bit32)src[1] << 8) |
                           ((bit32)src[2] << 16) | ((bit32)src[3] << 24);
                    interleaved[16 * batch_size * i + 4 * j + k] = word;
                } else {
                    interleaved[16 * batch_size * i + 4 * j + k] = 0;
                }
            }
        }
    }

    // 处理每个块
    for (int i = 0; i < max_blocks; ++i) {
        // 加载掩码
        __m128i mask = _mm_load_si128((__m128i*)&mask_array[i * batch_size]);

        // 加载消息字
        __m128i x[16];
        bit32* block_start = &interleaved[16 * batch_size * i];
        for (int j = 0; j < 16; j += 2) {
            x[j] = _mm_load_si128((__m128i*)(block_start + 4 * j));
            x[j + 1] = _mm_load_si128((__m128i*)(block_start + 4 * (j + 1)));
        }

        // 初始化临时状态
        __m128i a = state_a, b = state_b, c = state_c, d = state_d;

        // Round 1
        FF_SIMD(a, b, c, d, x[0], s11, T[0]);
        FF_SIMD(d, a, b, c, x[1], s12, T[1]);
        FF_SIMD(c, d, a, b, x[2], s13, T[2]);
        FF_SIMD(b, c, d, a, x[3], s14, T[3]);
        FF_SIMD(a, b, c, d, x[4], s11, T[4]);
        FF_SIMD(d, a, b, c, x[5], s12, T[5]);
        FF_SIMD(c, d, a, b, x[6], s13, T[6]);
        FF_SIMD(b, c, d, a, x[7], s14, T[7]);
        FF_SIMD(a, b, c, d, x[8], s11, T[8]);
        FF_SIMD(d, a, b, c, x[9], s12, T[9]);
        FF_SIMD(c, d, a, b, x[10], s13, T[10]);
        FF_SIMD(b, c, d, a, x[11], s14, T[11]);
        FF_SIMD(a, b, c, d, x[12], s11, T[12]);
        FF_SIMD(d, a, b, c, x[13], s12, T[13]);
        FF_SIMD(c, d, a, b, x[14], s13, T[14]);
        FF_SIMD(b, c, d, a, x[15], s14, T[15]);

        // Round 2
        GG_SIMD(a, b, c, d, x[1], s21, T[16]);
        GG_SIMD(d, a, b, c, x[6], s22, T[17]);
        GG_SIMD(c, d, a, b, x[11], s23, T[18]);
        GG_SIMD(b, c, d, a, x[0], s24, T[19]);
        GG_SIMD(a, b, c, d, x[5], s21, T[20]);
        GG_SIMD(d, a, b, c, x[10], s22, T[21]);
        GG_SIMD(c, d, a, b, x[15], s23, T[22]);
        GG_SIMD(b, c, d, a, x[4], s24, T[23]);
        GG_SIMD(a, b, c, d, x[9], s21, T[24]);
        GG_SIMD(d, a, b, c, x[14], s22, T[25]);
        GG_SIMD(c, d, a, b, x[3], s23, T[26]);
        GG_SIMD(b, c, d, a, x[8], s24, T[27]);
        GG_SIMD(a, b, c, d, x[13], s21, T[28]);
        GG_SIMD(d, a, b, c, x[2], s22, T[29]);
        GG_SIMD(c, d, a, b, x[7], s23, T[30]);
        GG_SIMD(b, c, d, a, x[12], s24, T[31]);

        // Round 3
        HH_SIMD(a, b, c, d, x[5], s31, T[32]);
        HH_SIMD(d, a, b, c, x[8], s32, T[33]);
        HH_SIMD(c, d, a, b, x[11], s33, T[34]);
        HH_SIMD(b, c, d, a, x[14], s34, T[35]);
        HH_SIMD(a, b, c, d, x[1], s31, T[36]);
        HH_SIMD(d, a, b, c, x[4], s32, T[37]);
        HH_SIMD(c, d, a, b, x[7], s33, T[38]);
        HH_SIMD(b, c, d, a, x[10], s34, T[39]);
        HH_SIMD(a, b, c, d, x[13], s31, T[40]);
        HH_SIMD(d, a, b, c, x[0], s32, T[41]);
        HH_SIMD(c, d, a, b, x[3], s33, T[42]);
        HH_SIMD(b, c, d, a, x[6], s34, T[43]);
        HH_SIMD(a, b, c, d, x[9], s31, T[44]);
        HH_SIMD(d, a, b, c, x[12], s32, T[45]);
        HH_SIMD(c, d, a, b, x[15], s33, T[46]);
        HH_SIMD(b, c, d, a, x[2], s34, T[47]);

        // Round 4
        II_SIMD(a, b, c, d, x[0], s41, T[48]);
        II_SIMD(d, a, b, c, x[7], s42, T[49]);
        II_SIMD(c, d, a, b, x[14], s43, T[50]);
        II_SIMD(b, c, d, a, x[5], s44, T[51]);
        II_SIMD(a, b, c, d, x[12], s41, T[52]);
        II_SIMD(d, a, b, c, x[3], s42, T[53]);
        II_SIMD(c, d, a, b, x[10], s43, T[54]);
        II_SIMD(b, c, d, a, x[1], s44, T[55]);
        II_SIMD(a, b, c, d, x[8], s41, T[56]);
        II_SIMD(d, a, b, c, x[15], s42, T[57]);
        II_SIMD(c, d, a, b, x[6], s43, T[58]);
        II_SIMD(b, c, d, a, x[13], s44, T[59]);
        II_SIMD(a, b, c, d, x[4], s41, T[60]);
        II_SIMD(d, a, b, c, x[11], s42, T[61]);
        II_SIMD(c, d, a, b, x[2], s43, T[62]);
        II_SIMD(b, c, d, a, x[9], s44, T[63]);

        // 使用掩码更新状态，模拟 NEON 的 vbslq_u32
        state_a = _mm_add_epi32(state_a, _mm_and_si128(a, mask));
        state_b = _mm_add_epi32(state_b, _mm_and_si128(b, mask));
        state_c = _mm_add_epi32(state_c, _mm_and_si128(c, mask));
        state_d = _mm_add_epi32(state_d, _mm_and_si128(d, mask));
    }

    // 字节反转以匹配小端字节序
    state_a = ByteSwap(state_a);
    state_b = ByteSwap(state_b);
    state_c = ByteSwap(state_c);
    state_d = ByteSwap(state_d);

    // 存储结果
    alignas(16) bit32 temp[16];
    _mm_store_si128((__m128i*)temp, state_a);
    _mm_store_si128((__m128i*)(temp + 4), state_b);
    _mm_store_si128((__m128i*)(temp + 8), state_c);
    _mm_store_si128((__m128i*)(temp + 12), state_d);

    for (int k = 0; k < batch_size; ++k) {
        states[k][0] = temp[k];
        states[k][1] = temp[k + 4];
        states[k][2] = temp[k + 8];
        states[k][3] = temp[k + 12];
    }

    // 释放内存
    for (int k = 0; k < batch_size; ++k) {
        delete[] paddedMessages[k];
    }
    delete[] paddedMessages;
    delete[] messageLength;
}

// 示例使用
int main() {
    for(int k=0;k<10000;k++)
    {
        std::string inputs[4] = {
            "",
            "hello world",
            "!@#$%^&*()_+=-0987654321",
            "this is a long string for testing hash function. it should produce a unique hash value."
        };
        alignas(16) bit32 states[4][4];

        MD5Hash_SSE(inputs, states, 4);

        // 打印哈希值
        for (int i = 0; i < 4; ++i) {
            std::cout << "MD5 Hash for input " << i << ": ";
            for (int j = 0; j < 4; ++j) {
                std::cout << std::hex << std::setfill('0') << std::setw(8) << states[i][j];
            }
            std::cout << std::endl;
        }

    }


    return 0;
}
