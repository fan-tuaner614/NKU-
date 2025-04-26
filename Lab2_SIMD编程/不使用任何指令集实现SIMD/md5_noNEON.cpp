#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <arm_neon.h>
using namespace std;
using namespace chrono;
#include <stdlib.h>
#include <vector>

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(string input, int *n_byte)
{
    // 将输入的字符串转换为Byte为单位的数组
    Byte *blocks = (Byte *)input.c_str();
    int length = input.length();

    // 计算原始消息长度（以比特为单位）
    int bitLength = length * 8;

    // paddingBits: 原始消息需要的padding长度（以bit为单位）
    // 对于给定的消息，将其补齐至length%512==448为止
    // 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
    int paddingBits = bitLength % 512;
    if (paddingBits > 448)
    {
        paddingBits = 512 - (paddingBits - 448);
    }
    else if (paddingBits < 448)
    {
        paddingBits = 448 - paddingBits;
    }
    else if (paddingBits == 448)
    {
        paddingBits = 512;
    }

    // 原始消息需要的padding长度（以Byte为单位）
    int paddingBytes = paddingBits / 8;
    // 创建最终的字节数组
    // length + paddingBytes + 8:
    // 1. length为原始消息的长度（bits）
    // 2. paddingBytes为原始消息需要的padding长度（Bytes）
    // 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
    int paddedLength = length + paddingBytes + 8;
    Byte *paddedMessage = new Byte[paddedLength];

    // 复制原始消息
    memcpy(paddedMessage, blocks, length);

    // 添加填充字节。填充时，第一位为1，后面的所有位均为0。
    // 所以第一个byte是0x80
    paddedMessage[length] = 0x80;                            // 添加一个0x80字节
    memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

    // 添加消息长度（64比特，小端格式）
    for (int i = 0; i < 8; ++i)
    {
        // 特别注意此处应当将bitLength转换为uint64_t
        // 这里的length是原始消息的长度
        paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
    }

    // 验证长度是否满足要求。此时长度应当是512bit的倍数
    int residual = 8 * paddedLength % 512;
    // assert(residual == 0);

    // 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
    *n_byte = paddedLength;
    return paddedMessage;
}

typedef unsigned char Byte;
typedef uint32_t bit32;
/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(string input, bit32 *state)
{

    Byte *paddedMessage;
    int *messageLength = new int[1];
    for (int i = 0; i < 1; i += 1)
    {
        paddedMessage = StringProcess(input, &messageLength[i]);
        // cout<<messageLength[i]<<endl;
        assert(messageLength[i] == messageLength[0]);
    }
    int n_blocks = messageLength[0] / 64;

    // bit32* state= new bit32[4];
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    // 逐block地更新state
    for (int i = 0; i < n_blocks; i += 1)
    {
        bit32 x[16];

        // 下面的处理，在理解上较为复杂
        for (int i1 = 0; i1 < 16; ++i1)
        {
            x[i1] = (paddedMessage[4 * i1 + i * 64]) |
                    (paddedMessage[4 * i1 + 1 + i * 64] << 8) |
                    (paddedMessage[4 * i1 + 2 + i * 64] << 16) |
                    (paddedMessage[4 * i1 + 3 + i * 64] << 24);
        }

        bit32 a = state[0], b = state[1], c = state[2], d = state[3];

        auto start = system_clock::now();
        /* Round 1 */
        FF(a, b, c, d, x[0], s11, 0xd76aa478);
        FF(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF(c, d, a, b, x[2], s13, 0x242070db);
        FF(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF(a, b, c, d, x[4], s11, 0xf57c0faf);
        FF(d, a, b, c, x[5], s12, 0x4787c62a);
        FF(c, d, a, b, x[6], s13, 0xa8304613);
        FF(b, c, d, a, x[7], s14, 0xfd469501);
        FF(a, b, c, d, x[8], s11, 0x698098d8);
        FF(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF(c, d, a, b, x[10], s13, 0xffff5bb1);
        FF(b, c, d, a, x[11], s14, 0x895cd7be);
        FF(a, b, c, d, x[12], s11, 0x6b901122);
        FF(d, a, b, c, x[13], s12, 0xfd987193);
        FF(c, d, a, b, x[14], s13, 0xa679438e);
        FF(b, c, d, a, x[15], s14, 0x49b40821);

        /* Round 2 */
        GG(a, b, c, d, x[1], s21, 0xf61e2562);
        GG(d, a, b, c, x[6], s22, 0xc040b340);
        GG(c, d, a, b, x[11], s23, 0x265e5a51);
        GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG(a, b, c, d, x[5], s21, 0xd62f105d);
        GG(d, a, b, c, x[10], s22, 0x2441453);
        GG(c, d, a, b, x[15], s23, 0xd8a1e681);
        GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG(a, b, c, d, x[9], s21, 0x21e1cde6);
        GG(d, a, b, c, x[14], s22, 0xc33707d6);
        GG(c, d, a, b, x[3], s23, 0xf4d50d87);
        GG(b, c, d, a, x[8], s24, 0x455a14ed);
        GG(a, b, c, d, x[13], s21, 0xa9e3e905);
        GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG(c, d, a, b, x[7], s23, 0x676f02d9);
        GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

        /* Round 3 */
        HH(a, b, c, d, x[5], s31, 0xfffa3942);
        HH(d, a, b, c, x[8], s32, 0x8771f681);
        HH(c, d, a, b, x[11], s33, 0x6d9d6122);
        HH(b, c, d, a, x[14], s34, 0xfde5380c);
        HH(a, b, c, d, x[1], s31, 0xa4beea44);
        HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
        HH(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH(a, b, c, d, x[13], s31, 0x289b7ec6);
        HH(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH(c, d, a, b, x[3], s33, 0xd4ef3085);
        HH(b, c, d, a, x[6], s34, 0x4881d05);
        HH(a, b, c, d, x[9], s31, 0xd9d4d039);
        HH(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
        HH(b, c, d, a, x[2], s34, 0xc4ac5665);

        /* Round 4 */
        II(a, b, c, d, x[0], s41, 0xf4292244);
        II(d, a, b, c, x[7], s42, 0x432aff97);
        II(c, d, a, b, x[14], s43, 0xab9423a7);
        II(b, c, d, a, x[5], s44, 0xfc93a039);
        II(a, b, c, d, x[12], s41, 0x655b59c3);
        II(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II(c, d, a, b, x[10], s43, 0xffeff47d);
        II(b, c, d, a, x[1], s44, 0x85845dd1);
        II(a, b, c, d, x[8], s41, 0x6fa87e4f);
        II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II(c, d, a, b, x[6], s43, 0xa3014314);
        II(b, c, d, a, x[13], s44, 0x4e0811a1);
        II(a, b, c, d, x[4], s41, 0xf7537e82);
        II(d, a, b, c, x[11], s42, 0xbd3af235);
        II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
        II(b, c, d, a, x[9], s44, 0xeb86d391);

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }

    // 下面的处理，在理解上较为复杂
    for (int i = 0; i < 4; i++)
    {
        uint32_t value = state[i];
        state[i] = ((value & 0xff) << 24) |      // 将最低字节移到最高位
                   ((value & 0xff00) << 8) |     // 将次低字节左移
                   ((value & 0xff0000) >> 8) |   // 将次高字节右移
                   ((value & 0xff000000) >> 24); // 将最高字节移到最低位
    }

    // 输出最终的hash结果
    // for (int i1 = 0; i1 < 4; i1 += 1)
    // {
    // 	cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    // }
    // cout << endl;

    // 释放动态分配的内存
    // 实现SIMD并行算法的时候，也请记得及时回收内存！
    delete[] paddedMessage;
    delete[] messageLength;
}

void MD5Hash_SIMD_noNEON(std::string inputs[4], bit32 states[4][4])
{
    const int batch_size = 4;
    assert(batch_size == 4 && "MD5Hash_SIMD_noNEON requires batch_size=4");

    // Initialize state vectors
    MD5Vector state_a(0x67452301, 0x67452301, 0x67452301, 0x67452301);
    MD5Vector state_b(0xefcdab89, 0xefcdab89, 0xefcdab89, 0xefcdab89);
    MD5Vector state_c(0x98badcfe, 0x98badcfe, 0x98badcfe, 0x98badcfe);
    MD5Vector state_d(0x10325476, 0x10325476, 0x10325476, 0x10325476);

    // Process input strings
    Byte **paddedMessages = new Byte *[batch_size];
    int *messageLength = new int[batch_size];
    int max_blocks = 0;
    for (int k = 0; k < batch_size; ++k)
    {
        paddedMessages[k] = StringProcess(inputs[k], &messageLength[k]);
        int n_blocks_k = messageLength[k] / 64;
        if (n_blocks_k > max_blocks)
            max_blocks = n_blocks_k;
    }

    // Create mask array, matching MD5Hash_4SIMD's batch generation
    alignas(16) bit32 mask_array[1024 * 4];
    for (int i = 0; i < max_blocks; i += 4)
    {
        for (int k = 0; k < batch_size; ++k)
        {
            mask_array[i * batch_size + k] = (i < messageLength[k] / 64) ? 0xFFFFFFFF : 0;
            mask_array[(i + 1) * batch_size + k] = ((i + 1) < messageLength[k] / 64) ? 0xFFFFFFFF : 0;
            mask_array[(i + 2) * batch_size + k] = ((i + 2) < messageLength[k] / 64) ? 0xFFFFFFFF : 0;
            mask_array[(i + 3) * batch_size + k] = ((i + 3) < messageLength[k] / 64) ? 0xFFFFFFFF : 0;
        }
    }

    // Interleave message blocks, matching MD5Hash_4SIMD's logic
    alignas(16) bit32 interleaved[16 * 1024 * 4];
    for (int i = 0; i < max_blocks; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            for (int k = 0; k < batch_size; ++k)
            {
                if (i < messageLength[k] / 64)
                {
                    bit32 word;
                    memcpy(&word, &paddedMessages[k][i * 64 + j * 4], sizeof(bit32));
                    interleaved[16 * batch_size * i + 4 * j + k] = word;
                }
                else
                {
                    interleaved[16 * batch_size * i + 4 * j + k] = 0;
                }
            }
        }
    }

    // Process blocks
    for (int i = 0; i < max_blocks; ++i)
    {
        // Load mask
        bit32 mask[4];
        for (int k = 0; k < batch_size; ++k)
        {
            mask[k] = mask_array[i * batch_size + k];
        }

        // Load message words
        MD5Vector x[16];
        bit32 *block_start = &interleaved[16 * batch_size * i];
        for (int j = 0; j < 16; ++j)
        {
            bit32 words[4];
            for (int k = 0; k < batch_size; ++k)
            {
                words[k] = block_start[4 * j + k];
            }
            x[j] = MD5Vector(words[0], words[1], words[2], words[3]);
        }

        // Initialize temporary state
        MD5Vector a = state_a, b = state_b, c = state_c, d = state_d;

        // Round 1
        FF_SIMD_noNEON(a, b, c, d, x[0], 7, 0xd76aa478);
        FF_SIMD_noNEON(d, a, b, c, x[1], 12, 0xe8c7b756);
        FF_SIMD_noNEON(c, d, a, b, x[2], 17, 0x242070db);
        FF_SIMD_noNEON(b, c, d, a, x[3], 22, 0xc1bdceee);
        FF_SIMD_noNEON(a, b, c, d, x[4], 7, 0xf57c0faf);
        FF_SIMD_noNEON(d, a, b, c, x[5], 12, 0x4787c62a);
        FF_SIMD_noNEON(c, d, a, b, x[6], 17, 0xa8304613);
        FF_SIMD_noNEON(b, c, d, a, x[7], 22, 0xfd469501);
        FF_SIMD_noNEON(a, b, c, d, x[8], 7, 0x698098d8);
        FF_SIMD_noNEON(d, a, b, c, x[9], 12, 0x8b44f7af);
        FF_SIMD_noNEON(c, d, a, b, x[10], 17, 0xffff5bb1);
        FF_SIMD_noNEON(b, c, d, a, x[11], 22, 0x895cd7be);
        FF_SIMD_noNEON(a, b, c, d, x[12], 7, 0x6b901122);
        FF_SIMD_noNEON(d, a, b, c, x[13], 12, 0xfd987193);
        FF_SIMD_noNEON(c, d, a, b, x[14], 17, 0xa679438e);
        FF_SIMD_noNEON(b, c, d, a, x[15], 22, 0x49b40821);

        // Round 2
        GG_SIMD_noNEON(a, b, c, d, x[1], 5, 0xf61e2562);
        GG_SIMD_noNEON(d, a, b, c, x[6], 9, 0xc040b340);
        GG_SIMD_noNEON(c, d, a, b, x[11], 14, 0x265e5a51);
        GG_SIMD_noNEON(b, c, d, a, x[0], 20, 0xe9b6c7aa);
        GG_SIMD_noNEON(a, b, c, d, x[5], 5, 0xd62f105d);
        GG_SIMD_noNEON(d, a, b, c, x[10], 9, 0x02441453);
        GG_SIMD_noNEON(c, d, a, b, x[15], 14, 0xd8a1e681);
        GG_SIMD_noNEON(b, c, d, a, x[4], 20, 0xe7d3fbc8);
        GG_SIMD_noNEON(a, b, c, d, x[9], 5, 0x21e1cde6);
        GG_SIMD_noNEON(d, a, b, c, x[14], 9, 0xc33707d6);
        GG_SIMD_noNEON(c, d, a, b, x[3], 14, 0xf4d50d87);
        GG_SIMD_noNEON(b, c, d, a, x[8], 20, 0x455a14ed);
        GG_SIMD_noNEON(a, b, c, d, x[13], 5, 0xa9e3e905);
        GG_SIMD_noNEON(d, a, b, c, x[2], 9, 0xfcefa3f8);
        GG_SIMD_noNEON(c, d, a, b, x[7], 14, 0x676f02d9);
        GG_SIMD_noNEON(b, c, d, a, x[12], 20, 0x8d2a4c8a);

        // Round 3
        HH_SIMD_noNEON(a, b, c, d, x[5], 4, 0xfffa3942);
        HH_SIMD_noNEON(d, a, b, c, x[8], 11, 0x8771f681);
        HH_SIMD_noNEON(c, d, a, b, x[11], 16, 0x6d9d6122);
        HH_SIMD_noNEON(b, c, d, a, x[14], 23, 0xfde5380c);
        HH_SIMD_noNEON(a, b, c, d, x[1], 4, 0xa4beea44);
        HH_SIMD_noNEON(d, a, b, c, x[4], 11, 0x4bdecfa9);
        HH_SIMD_noNEON(c, d, a, b, x[7], 16, 0xf6bb4b60);
        HH_SIMD_noNEON(b, c, d, a, x[10], 23, 0xbebfbc70);
        HH_SIMD_noNEON(a, b, c, d, x[13], 4, 0x289b7ec6);
        HH_SIMD_noNEON(d, a, b, c, x[0], 11, 0xeaa127fa);
        HH_SIMD_noNEON(c, d, a, b, x[3], 16, 0xd4ef3085);
        HH_SIMD_noNEON(b, c, d, a, x[6], 23, 0x04881d05);
        HH_SIMD_noNEON(a, b, c, d, x[9], 4, 0xd9d4d039);
        HH_SIMD_noNEON(d, a, b, c, x[12], 11, 0xe6db99e5);
        HH_SIMD_noNEON(c, d, a, b, x[15], 16, 0x1fa27cf8);
        HH_SIMD_noNEON(b, c, d, a, x[2], 23, 0xc4ac5665);

        // Round 4
        II_SIMD_noNEON(a, b, c, d, x[0], 6, 0xf4292244);
        II_SIMD_noNEON(d, a, b, c, x[7], 10, 0x432aff97);
        II_SIMD_noNEON(c, d, a, b, x[14], 15, 0xab9423a7);
        II_SIMD_noNEON(b, c, d, a, x[5], 21, 0xfc93a039);
        II_SIMD_noNEON(a, b, c, d, x[12], 6, 0x655b59c3);
        II_SIMD_noNEON(d, a, b, c, x[3], 10, 0x8f0ccc92);
        II_SIMD_noNEON(c, d, a, b, x[10], 15, 0xffeff47d);
        II_SIMD_noNEON(b, c, d, a, x[1], 21, 0x85845dd1);
        II_SIMD_noNEON(a, b, c, d, x[8], 6, 0x6fa87e4f);
        II_SIMD_noNEON(d, a, b, c, x[15], 10, 0xfe2ce6e0);
        II_SIMD_noNEON(c, d, a, b, x[6], 15, 0xa3014314);
        II_SIMD_noNEON(b, c, d, a, x[13], 21, 0x4e0811a1);
        II_SIMD_noNEON(a, b, c, d, x[4], 6, 0xf7537e82);
        II_SIMD_noNEON(d, a, b, c, x[11], 10, 0xbd3af235);
        II_SIMD_noNEON(c, d, a, b, x[2], 15, 0x2ad7d2bb);
        II_SIMD_noNEON(b, c, d, a, x[9], 21, 0xeb86d391);

        // Update state with mask
        for (int k = 0; k < 4; ++k)
        {
            bit32 m = mask[k];
            state_a.v[k] += (m == 0xFFFFFFFF) ? a.v[k] : 0;
            state_b.v[k] += (m == 0xFFFFFFFF) ? b.v[k] : 0;
            state_c.v[k] += (m == 0xFFFFFFFF) ? c.v[k] : 0;
            state_d.v[k] += (m == 0xFFFFFFFF) ? d.v[k] : 0;
        }
    }

    // Byte swapping to match MD5Hash_4SIMD
    state_a = ByteSwap(state_a);
    state_b = ByteSwap(state_b);
    state_c = ByteSwap(state_c);
    state_d = ByteSwap(state_d);

    // Store results
    alignas(16) bit32 temp[16];
    for (int k = 0; k < 4; ++k)
    {
        temp[k] = state_a.v[k];
        temp[k + 4] = state_b.v[k];
        temp[k + 8] = state_c.v[k];
        temp[k + 12] = state_d.v[k];
    }

    for (int k = 0; k < batch_size; ++k)
    {
        states[k][0] = temp[k];
        states[k][1] = temp[k + 4];
        states[k][2] = temp[k + 8];
        states[k][3] = temp[k + 12];
    }

    // Clean up
    for (int k = 0; k < batch_size; ++k)
    {
        delete[] paddedMessages[k];
    }
    delete[] paddedMessages;
    delete[] messageLength;
}