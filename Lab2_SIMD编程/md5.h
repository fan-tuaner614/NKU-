#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>
using namespace std;
#include <array>
#include <cstdint>

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21
struct MD5Vector
{
    bit32 v[4];
    MD5Vector() {}
    MD5Vector(bit32 v0, bit32 v1, bit32 v2, bit32 v3)
    {
        v[0] = v0;
        v[1] = v1;
        v[2] = v2;
        v[3] = v3;
    }
    MD5Vector operator+(const MD5Vector &other) const
    {
        return {v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2], v[3] + other.v[3]};
    }
    MD5Vector operator&(const MD5Vector &other) const
    {
        return {v[0] & other.v[0], v[1] & other.v[1], v[2] & other.v[2], v[3] & other.v[3]};
    }
    MD5Vector operator|(const MD5Vector &other) const
    {
        return {v[0] | other.v[0], v[1] | other.v[1], v[2] | other.v[2], v[3] | other.v[3]};
    }
    MD5Vector operator^(const MD5Vector &other) const
    {
        return {v[0] ^ other.v[0], v[1] ^ other.v[1], v[2] ^ other.v[2], v[3] ^ other.v[3]};
    }
    MD5Vector operator~() const
    {
        return {~v[0], ~v[1], ~v[2], ~v[3]};
    }
};

inline MD5Vector RotateLeft(const MD5Vector &vec, int s)
{
    MD5Vector result;
    for (int i = 0; i < 4; ++i)
    {
        result.v[i] = (vec.v[i] << s) | (vec.v[i] >> (32 - s));
    }
    return result;
}
inline MD5Vector ByteSwap(const MD5Vector &vec)
{
    MD5Vector result;
    for (int i = 0; i < 4; ++i)
    {
        bit32 val = vec.v[i];
        result.v[i] = ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) |
                      ((val & 0xFF0000) >> 8) | ((val & 0xFF000000) >> 24);
    }
    return result;
}
inline void FF_SIMD_noNEON(MD5Vector &a, MD5Vector b, MD5Vector c, MD5Vector d, MD5Vector x, int s, bit32 ac)
{
    MD5Vector f = (b & c) | (~b & d);
    MD5Vector temp = a + f + x + MD5Vector(ac, ac, ac, ac);
    temp = RotateLeft(temp, s);
    a = b + temp;
}

inline void GG_SIMD_noNEON(MD5Vector &a, MD5Vector b, MD5Vector c, MD5Vector d, MD5Vector x, int s, bit32 ac)
{
    MD5Vector g = (b & d) | (c & ~d);
    MD5Vector temp = a + g + x + MD5Vector(ac, ac, ac, ac);
    temp = RotateLeft(temp, s);
    a = b + temp;
}

inline void HH_SIMD_noNEON(MD5Vector &a, MD5Vector b, MD5Vector c, MD5Vector d, MD5Vector x, int s, bit32 ac)
{
    MD5Vector h = b ^ c ^ d;
    MD5Vector temp = a + h + x + MD5Vector(ac, ac, ac, ac);
    temp = RotateLeft(temp, s);
    a = b + temp;
}

inline void II_SIMD_noNEON(MD5Vector &a, MD5Vector b, MD5Vector c, MD5Vector d, MD5Vector x, int s, bit32 ac)
{
    MD5Vector i = c ^ (b | ~d);
    MD5Vector temp = a + i + x + MD5Vector(ac, ac, ac, ac);
    temp = RotateLeft(temp, s);
    a = b + temp;
}
/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
// 定义了一系列MD5中的具体函数
// 这四个计算函数是需要你进行SIMD并行化的
// 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化
// 模拟4路并行计算的向量结构
// 宏定义 FF_NEON，嵌入 F 逻辑
#define FF_SIMD(a, b, c, d, x, s, ac)                        \
    do                                                       \
    {                                                        \
        uint32x4_t ac_vec = vdupq_n_u32(ac);                 \
        uint32x4_t xy = vandq_u32(b, c);                     \
        uint32x4_t notx_z = vandq_u32(vmvnq_u32(b), d);      \
        uint32x4_t f = vorrq_u32(xy, notx_z);                \
        a = vaddq_u32(a, f);                                 \
        a = vaddq_u32(a, x);                                 \
        a = vaddq_u32(a, ac_vec);                            \
        a = vorrq_u32(vshlq_u32(a, vdupq_n_s32(s)),          \
                      vshlq_u32(a, vdupq_n_s32(-(32 - s)))); \
        a = vaddq_u32(a, b);                                 \
    } while (0)

// 宏定义 GG_SIMD，嵌入 G 逻辑
#define GG_SIMD(a, b, c, d, x, s, ac)                        \
    do                                                       \
    {                                                        \
        uint32x4_t ac_vec = vdupq_n_u32(ac);                 \
        uint32x4_t xz = vandq_u32(b, d);                     \
        uint32x4_t y_nz = vandq_u32(c, vmvnq_u32(d));        \
        uint32x4_t g = vorrq_u32(xz, y_nz);                  \
        a = vaddq_u32(a, g);                                 \
        a = vaddq_u32(a, x);                                 \
        a = vaddq_u32(a, ac_vec);                            \
        a = vorrq_u32(vshlq_u32(a, vdupq_n_s32(s)),          \
                      vshlq_u32(a, vdupq_n_s32(-(32 - s)))); \
        a = vaddq_u32(a, b);                                 \
    } while (0)

// 宏定义 HH_SIMD，嵌入 H 逻辑
#define HH_SIMD(a, b, c, d, x, s, ac)                        \
    do                                                       \
    {                                                        \
        uint32x4_t ac_vec = vdupq_n_u32(ac);                 \
        uint32x4_t h = veorq_u32(veorq_u32(b, c), d);        \
        a = vaddq_u32(a, h);                                 \
        a = vaddq_u32(a, x);                                 \
        a = vaddq_u32(a, ac_vec);                            \
        a = vorrq_u32(vshlq_u32(a, vdupq_n_s32(s)),          \
                      vshlq_u32(a, vdupq_n_s32(-(32 - s)))); \
        a = vaddq_u32(a, b);                                 \
    } while (0)

// 宏定义 II_SIMD，嵌入 I 逻辑
#define II_SIMD(a, b, c, d, x, s, ac)                        \
    do                                                       \
    {                                                        \
        uint32x4_t ac_vec = vdupq_n_u32(ac);                 \
        uint32x4_t not_z = vmvnq_u32(d);                     \
        uint32x4_t i = veorq_u32(c, vorrq_u32(b, not_z));    \
        a = vaddq_u32(a, i);                                 \
        a = vaddq_u32(a, x);                                 \
        a = vaddq_u32(a, ac_vec);                            \
        a = vorrq_u32(vshlq_u32(a, vdupq_n_s32(s)),          \
                      vshlq_u32(a, vdupq_n_s32(-(32 - s)))); \
        a = vaddq_u32(a, b);                                 \
    } while (0)
// 原始函数：F(x,y,z) = ((x & y) | (~x & z))
// 优化后的 FF_NEON，嵌入 F_neon 逻辑

// 并行计算4个F函数（同时处理4组输入）
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))
/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
// 定义了一系列MD5中的具体函数
// 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
// 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法
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
const int BATCH_SIZE = 4;
void MD5Hash_8SIMD(const string *inputs, uint32_t states[][4], int batch_size);
void MD5Hash_4SIMD(const string *inputs, uint32_t states[][4], int batch_size);
void MD5Hash_2SIMD(const string *inputs, uint32_t states[][4], int batch_size);
void MD5Hash_SIMD(const string *inputs, uint32_t states[][4], int batch_size);
void MD5Hash_SIMD_noNEON(string inputs[4], bit32 states[4][4]);
void MD5Hash(string input, bit32 *state);