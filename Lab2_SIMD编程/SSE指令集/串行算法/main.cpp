#include <cstring>
#include <string>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;
using Byte = unsigned char;
using bit32 = uint32_t;

// MD5 ��׼��ʼֵ
const bit32 INIT_A = 0x67452301;
const bit32 INIT_B = 0xefcdab89;
const bit32 INIT_C = 0x98badcfe;
const bit32 INIT_D = 0x10325476;

// ��λ����
const int s11 = 7, s12 = 12, s13 = 17, s14 = 22;
const int s21 = 5, s22 = 9, s23 = 14, s24 = 20;
const int s31 = 4, s32 = 11, s33 = 16, s34 = 23;
const int s41 = 6, s42 = 10, s43 = 15, s44 = 21;

// MD5 �ֺ����еĳ�����T��
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

// �궨�壨ֱ�Ӹ����ṩ�Ĵ��룩
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32 - (n))))

#define FF(a, b, c, d, x, s, ac)            \
    {                                       \
        (a) += F((b), (c), (d)) + (x) + ac; \
        (a) = ROTATELEFT((a), (s));         \
        (a) += (b);                         \
    }

#define GG(a, b, c, d, x, s, ac)            \
    {                                       \
        (a) += G((b), (c), (d)) + (x) + ac; \
        (a) = ROTATELEFT((a), (s));         \
        (a) += (b);                         \
    }

#define HH(a, b, c, d, x, s, ac)            \
    {                                       \
        (a) += H((b), (c), (d)) + (x) + ac; \
        (a) = ROTATELEFT((a), (s));         \
        (a) += (b);                         \
    }

#define II(a, b, c, d, x, s, ac)            \
    {                                       \
        (a) += I((b), (c), (d)) + (x) + ac; \
        (a) = ROTATELEFT((a), (s));         \
        (a) += (b);                         \
    }

// ��Ϣ��亯����ֱ�Ӹ����ṩ�Ĵ��룩
Byte *StringProcess(string input, int *n_byte)
{
    // ��������ַ���ת��ΪByteΪ��λ������
    Byte *blocks = (Byte *)input.c_str();
    int length = input.length();

    // ����ԭʼ��Ϣ���ȣ��Ա���Ϊ��λ��
    int bitLength = length * 8;

    // paddingBits: ԭʼ��Ϣ��Ҫ��padding���ȣ���bitΪ��λ��
    // ���ڸ�������Ϣ�����䲹����length%512==448Ϊֹ
    // ��Ҫע����ǣ������������Ϣ����length%512==448��Ҳ��Ҫ��pad 512bits
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

    // ԭʼ��Ϣ��Ҫ��padding���ȣ���ByteΪ��λ��
    int paddingBytes = paddingBits / 8;
    // �������յ��ֽ�����
    // length + paddingBytes + 8:
    // 1. lengthΪԭʼ��Ϣ�ĳ��ȣ�bits��
    // 2. paddingBytesΪԭʼ��Ϣ��Ҫ��padding���ȣ�Bytes��
    // 3. ��pad��length%512==448֮����Ҫ���⸽��64bits��ԭʼ��Ϣ���ȣ���8��bytes
    int paddedLength = length + paddingBytes + 8;
    Byte *paddedMessage = new Byte[paddedLength];

    // ����ԭʼ��Ϣ
    memcpy(paddedMessage, blocks, length);

    // �������ֽڡ����ʱ����һλΪ1�����������λ��Ϊ0��
    // ���Ե�һ��byte��0x80
    paddedMessage[length] = 0x80;                            // ���һ��0x80�ֽ�
    memset(paddedMessage + length + 1, 0, paddingBytes - 1); // ���0�ֽ�

    // �����Ϣ���ȣ�64���أ�С�˸�ʽ��
    for (int i = 0; i < 8; ++i)
    {
        // �ر�ע��˴�Ӧ����bitLengthת��Ϊuint64_t
        // �����length��ԭʼ��Ϣ�ĳ���
        paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
    }

    // ��֤�����Ƿ�����Ҫ�󡣴�ʱ����Ӧ����512bit�ı���
    int residual = 8 * paddedLength % 512;
    // assert(residual == 0);

    // �����+��ӳ���֮����Ϣ����Ϊn_blocks��512bit�Ĳ���
    *n_byte = paddedLength;
    return paddedMessage;
}

// ���� MD5 ��ϣ�����������ṩ�� MD5Hash ʵ�֣�
void MD5Hash(string input, bit32 *state)
{
    Byte *paddedMessage;
    int *messageLength = new int[1];
    for (int i = 0; i < 1; i += 1)
    {
        paddedMessage = StringProcess(input, &messageLength[i]);
        assert(messageLength[i] == messageLength[0]);
    }
    int n_blocks = messageLength[0] / 64;

    // ��ʼ��״̬
    state[0] = INIT_A;
    state[1] = INIT_B;
    state[2] = INIT_C;
    state[3] = INIT_D;

    // ������״̬
    for (int i = 0; i < n_blocks; i += 1)
    {
        bit32 x[16];

        // ���� 512 λ�飨64 �ֽڣ��� 16 �� 32 λ��
        for (int i1 = 0; i1 < 16; ++i1)
        {
            x[i1] = (paddedMessage[4 * i1 + i * 64]) |
                    (paddedMessage[4 * i1 + 1 + i * 64] << 8) |
                    (paddedMessage[4 * i1 + 2 + i * 64] << 16) |
                    (paddedMessage[4 * i1 + 3 + i * 64] << 24);
        }

        bit32 a = state[0], b = state[1], c = state[2], d = state[3];

        // Round 1
        FF(a, b, c, d, x[0], s11, T[0]);
        FF(d, a, b, c, x[1], s12, T[1]);
        FF(c, d, a, b, x[2], s13, T[2]);
        FF(b, c, d, a, x[3], s14, T[3]);
        FF(a, b, c, d, x[4], s11, T[4]);
        FF(d, a, b, c, x[5], s12, T[5]);
        FF(c, d, a, b, x[6], s13, T[6]);
        FF(b, c, d, a, x[7], s14, T[7]);
        FF(a, b, c, d, x[8], s11, T[8]);
        FF(d, a, b, c, x[9], s12, T[9]);
        FF(c, d, a, b, x[10], s13, T[10]);
        FF(b, c, d, a, x[11], s14, T[11]);
        FF(a, b, c, d, x[12], s11, T[12]);
        FF(d, a, b, c, x[13], s12, T[13]);
        FF(c, d, a, b, x[14], s13, T[14]);
        FF(b, c, d, a, x[15], s14, T[15]);

        // Round 2
        GG(a, b, c, d, x[1], s21, T[16]);
        GG(d, a, b, c, x[6], s22, T[17]);
        GG(c, d, a, b, x[11], s23, T[18]);
        GG(b, c, d, a, x[0], s24, T[19]);
        GG(a, b, c, d, x[5], s21, T[20]);
        GG(d, a, b, c, x[10], s22, T[21]);
        GG(c, d, a, b, x[15], s23, T[22]);
        GG(b, c, d, a, x[4], s24, T[23]);
        GG(a, b, c, d, x[9], s21, T[24]);
        GG(d, a, b, c, x[14], s22, T[25]);
        GG(c, d, a, b, x[3], s23, T[26]);
        GG(b, c, d, a, x[8], s24, T[27]);
        GG(a, b, c, d, x[13], s21, T[28]);
        GG(d, a, b, c, x[2], s22, T[29]);
        GG(c, d, a, b, x[7], s23, T[30]);
        GG(b, c, d, a, x[12], s24, T[31]);

        // Round 3
        HH(a, b, c, d, x[5], s31, T[32]);
        HH(d, a, b, c, x[8], s32, T[33]);
        HH(c, d, a, b, x[11], s33, T[34]);
        HH(b, c, d, a, x[14], s34, T[35]);
        HH(a, b, c, d, x[1], s31, T[36]);
        HH(d, a, b, c, x[4], s32, T[37]);
        HH(c, d, a, b, x[7], s33, T[38]);
        HH(b, c, d, a, x[10], s34, T[39]);
        HH(a, b, c, d, x[13], s31, T[40]);
        HH(d, a, b, c, x[0], s32, T[41]);
        HH(c, d, a, b, x[3], s33, T[42]);
        HH(b, c, d, a, x[6], s34, T[43]);
        HH(a, b, c, d, x[9], s31, T[44]);
        HH(d, a, b, c, x[12], s32, T[45]);
        HH(c, d, a, b, x[15], s33, T[46]);
        HH(b, c, d, a, x[2], s34, T[47]);

        // Round 4
        II(a, b, c, d, x[0], s41, T[48]);
        II(d, a, b, c, x[7], s42, T[49]);
        II(c, d, a, b, x[14], s43, T[50]);
        II(b, c, d, a, x[5], s44, T[51]);
        II(a, b, c, d, x[12], s41, T[52]);
        II(d, a, b, c, x[3], s42, T[53]);
        II(c, d, a, b, x[10], s43, T[54]);
        II(b, c, d, a, x[1], s44, T[55]);
        II(a, b, c, d, x[8], s41, T[56]);
        II(d, a, b, c, x[15], s42, T[57]);
        II(c, d, a, b, x[6], s43, T[58]);
        II(b, c, d, a, x[13], s44, T[59]);
        II(a, b, c, d, x[4], s41, T[60]);
        II(d, a, b, c, x[11], s42, T[61]);
        II(c, d, a, b, x[2], s43, T[62]);
        II(b, c, d, a, x[9], s44, T[63]);

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }

    // �ֽڷ�ת��ƥ���׼ MD5 �����ʽ
    for (int i = 0; i < 4; i++)
    {
        uint32_t value = state[i];
        state[i] = ((value & 0xff) << 24) |
                   ((value & 0xff00) << 8) |
                   ((value & 0xff0000) >> 8) |
                   ((value & 0xff000000) >> 24);
    }

    // �ͷ��ڴ�
    delete[] paddedMessage;
    delete[] messageLength;
}

// �����������ڲ��� MD5 ��ϣ�㷨
int main()
{
    for(int k=0;k<10000;k++)
    {
            // ��������
        string inputs[4] = {
            "",
            "hello world",
            "!@#$%^&*()_+=-0987654321",
            "this is a long string for testing hash function. it should produce a unique hash value."
        };

        // ���㲢�����ϣֵ
        for (int i = 0; i < 4; ++i)
        {
            bit32 state[4];
            MD5Hash(inputs[i], state);
            cout << "MD5 Hash for input " << i << ": ";
            for (int j = 0; j < 4; ++j)
            {
                cout << hex << setw(8) << setfill('0') << state[j];
            }
            cout << endl;
        }
    }
    return 0;
}
