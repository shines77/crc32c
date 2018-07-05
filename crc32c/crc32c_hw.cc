
#include "logging/crc32c.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>

#ifdef __SSE4_2__
#ifdef _MSC_VER
#include <nmmintrin.h>  // For SSE 4.2
#include <wnmmintrin.h>  // For SSE 4.2
#else
#include <x86intrin.h>
//#include <nmmintrin.h>  // For SSE 4.2
#endif // _MSC_VER
#endif // __SSE4_2__

#ifndef ssize_t
#define ssize_t ptrdiff_t
#endif

#if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
 || defined(__amd64__) || defined(__x86_64__) || defined(__LP64__)
#ifndef CRC32C_IS_X86_64
#define CRC32C_IS_X86_64     1
#endif
#endif // _WIN64 || __amd64__

#ifndef __has_attribute
#define CRC32C_HAS_ATTRIBUTE(x)     0
#else
#define CRC32C_HAS_ATTRIBUTE(x)     __has_attribute(x)
#endif

#ifndef __has_cpp_attribute
#define CRC32C_HAS_CPP_ATTRIBUTE(x) 0
#else
#define CRC32C_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#endif

#ifndef __has_extension
#define CRC32C_HAS_EXTENSION(x)     0
#else
#define CRC32C_HAS_EXTENSION(x)     __has_extension(x)
#endif

/**
 * Fall-through to indicate that `break` was left out on purpose in a switch
 * statement, e.g.
 *
 * switch (n) {
 *   case 22:
 *   case 33:  // no warning: no statements between case labels
 *     f();
 *   case 44:  // warning: unannotated fall-through
 *     g();
 *     FOLLY_FALLTHROUGH; // no warning: annotated fall-through
 * }
 */
#if CRC32C_HAS_CPP_ATTRIBUTE(fallthrough)
  #define CRC32C_FALLTHROUGH    [[fallthrough]]
#elif CRC32C_HAS_CPP_ATTRIBUTE(clang::fallthrough)
  #define CRC32C_FALLTHROUGH    [[clang::fallthrough]]
#elif CRC32C_HAS_CPP_ATTRIBUTE(gnu::fallthrough)
  #define CRC32C_FALLTHROUGH    [[gnu::fallthrough]]
#else
  #define CRC32C_FALLTHROUGH
#endif

#if CRC32C_IS_X86_64

#define CRC32C_Triplet(crc, buf, offset)                                        \
    do {                                                                        \
        crc##0 = _mm_crc32_u64(crc##0, *((uint64_t *)(buf##0) + (offset)));     \
        crc##1 = _mm_crc32_u64(crc##1, *((uint64_t *)(buf##1) + (offset)));     \
        crc##2 = _mm_crc32_u64(crc##2, *((uint64_t *)(buf##2) + (offset)));     \
    } while (0);                                                                \
    CRC32C_FALLTHROUGH

#define CRC32C_Duplet(crc, buf, offset)                                         \
    do {                                                                        \
        crc##0 = _mm_crc32_u64(crc##0, *((uint64_t *)(buf##0) + (offset)));     \
        crc##1 = _mm_crc32_u64(crc##1, *((uint64_t *)(buf##1) + (offset)));     \
    } while (0);                                                                \
    CRC32C_FALLTHROUGH

#define CRC32C_Singlet(crc, buf, offset)                                        \
    do {                                                                        \
        crc = _mm_crc32_u64(crc, *((uint64_t *)(buf) + offset));                \
    } while (0);                                                                \
    CRC32C_FALLTHROUGH

#else // !CRC32C_IS_X86_64

#define CRC32C_Triplet(crc, buf, offset)                                                \
    do {                                                                                \
        crc##0 = _mm_crc32_u32(crc##0, *((uint32_t *)(buf##0) + 0 + 2 * (offset)));     \
        crc##1 = _mm_crc32_u32(crc##1, *((uint32_t *)(buf##1) + 0 + 2 * (offset)));     \
        crc##2 = _mm_crc32_u32(crc##2, *((uint32_t *)(buf##2) + 0 + 2 * (offset)));     \
        crc##0 = _mm_crc32_u32(crc##0, *((uint32_t *)(buf##0) + 1 + 2 * (offset)));     \
        crc##1 = _mm_crc32_u32(crc##1, *((uint32_t *)(buf##1) + 1 + 2 * (offset)));     \
        crc##2 = _mm_crc32_u32(crc##2, *((uint32_t *)(buf##2) + 1 + 2 * (offset)));     \
    } while (0);                                                                        \
    CRC32C_FALLTHROUGH

#define CRC32C_Duplet(crc, buf, offset)                                                 \
    do {                                                                                \
        crc##0 = _mm_crc32_u32(crc##0, *((uint32_t *)(buf##0) + 0 + 2 * (offset)));     \
        crc##1 = _mm_crc32_u32(crc##1, *((uint32_t *)(buf##1) + 0 + 2 * (offset)));     \
        crc##0 = _mm_crc32_u32(crc##0, *((uint32_t *)(buf##0) + 1 + 2 * (offset)));     \
        crc##1 = _mm_crc32_u32(crc##1, *((uint32_t *)(buf##1) + 1 + 2 * (offset)));     \
    } while (0);                                                                        \
    CRC32C_FALLTHROUGH

#define CRC32C_Singlet(crc, buf, offset)                                                \
    do {                                                                                \
        crc = _mm_crc32_u32(crc, *((uint32_t *)(buf) + 0 + 2 * (offset)));              \
        crc = _mm_crc32_u32(crc, *((uint32_t *)(buf) + 1 + 2 * (offset)));              \
    } while (0);                                                                        \
    CRC32C_FALLTHROUGH

#endif // CRC32C_IS_X86_64

namespace logging {

//
// Numbers taken directly from intel white-paper.
//
// clang-format off
//alignas(16)
static const uint64_t crc32c_clmul_constants[] = {
    0x14cd00bd6ULL, 0x105ec76f0ULL, 0x0ba4fc28eULL, 0x14cd00bd6ULL,
    0x1d82c63daULL, 0x0f20c0dfeULL, 0x09e4addf8ULL, 0x0ba4fc28eULL,
    0x039d3b296ULL, 0x1384aa63aULL, 0x102f9b8a2ULL, 0x1d82c63daULL,
    0x14237f5e6ULL, 0x01c291d04ULL, 0x00d3b6092ULL, 0x09e4addf8ULL,
    0x0c96cfdc0ULL, 0x0740eef02ULL, 0x18266e456ULL, 0x039d3b296ULL,
    0x0daece73eULL, 0x0083a6eecULL, 0x0ab7aff2aULL, 0x102f9b8a2ULL,
    0x1248ea574ULL, 0x1c1733996ULL, 0x083348832ULL, 0x14237f5e6ULL,
    0x12c743124ULL, 0x02ad91c30ULL, 0x0b9e02b86ULL, 0x00d3b6092ULL,
    0x018b33a4eULL, 0x06992cea2ULL, 0x1b331e26aULL, 0x0c96cfdc0ULL,
    0x17d35ba46ULL, 0x07e908048ULL, 0x1bf2e8b8aULL, 0x18266e456ULL,
    0x1a3e0968aULL, 0x11ed1f9d8ULL, 0x0ce7f39f4ULL, 0x0daece73eULL,
    0x061d82e56ULL, 0x0f1d0f55eULL, 0x0d270f1a2ULL, 0x0ab7aff2aULL,
    0x1c3f5f66cULL, 0x0a87ab8a8ULL, 0x12ed0daacULL, 0x1248ea574ULL,
    0x065863b64ULL, 0x08462d800ULL, 0x11eef4f8eULL, 0x083348832ULL,
    0x1ee54f54cULL, 0x071d111a8ULL, 0x0b3e32c28ULL, 0x12c743124ULL,
    0x0064f7f26ULL, 0x0ffd852c6ULL, 0x0dd7e3b0cULL, 0x0b9e02b86ULL,
    0x0f285651cULL, 0x0dcb17aa4ULL, 0x010746f3cULL, 0x018b33a4eULL,
    0x1c24afea4ULL, 0x0f37c5aeeULL, 0x0271d9844ULL, 0x1b331e26aULL,
    0x08e766a0cULL, 0x06051d5a2ULL, 0x093a5f730ULL, 0x17d35ba46ULL,
    0x06cb08e5cULL, 0x11d5ca20eULL, 0x06b749fb2ULL, 0x1bf2e8b8aULL,
    0x1167f94f2ULL, 0x021f3d99cULL, 0x0cec3662eULL, 0x1a3e0968aULL,
    0x19329634aULL, 0x08f158014ULL, 0x0e6fc4e6aULL, 0x0ce7f39f4ULL,
    0x08227bb8aULL, 0x1a5e82106ULL, 0x0b0cd4768ULL, 0x061d82e56ULL,
    0x13c2b89c4ULL, 0x188815ab2ULL, 0x0d7a4825cULL, 0x0d270f1a2ULL,
    0x10f5ff2baULL, 0x105405f3eULL, 0x00167d312ULL, 0x1c3f5f66cULL,
    0x0f6076544ULL, 0x0e9adf796ULL, 0x026f6a60aULL, 0x12ed0daacULL,
    0x1a2adb74eULL, 0x096638b34ULL, 0x19d34af3aULL, 0x065863b64ULL,
    0x049c3cc9cULL, 0x1e50585a0ULL, 0x068bce87aULL, 0x11eef4f8eULL,
    0x1524fa6c6ULL, 0x19f1c69dcULL, 0x16cba8acaULL, 0x1ee54f54cULL,
    0x042d98888ULL, 0x12913343eULL, 0x1329d9f7eULL, 0x0b3e32c28ULL,
    0x1b1c69528ULL, 0x088f25a3aULL, 0x02178513aULL, 0x0064f7f26ULL,
    0x0e0ac139eULL, 0x04e36f0b0ULL, 0x0170076faULL, 0x0dd7e3b0cULL,
    0x141a1a2e2ULL, 0x0bd6f81f8ULL, 0x16ad828b4ULL, 0x0f285651cULL,
    0x041d17b64ULL, 0x19425cbbaULL, 0x1fae1cc66ULL, 0x010746f3cULL,
    0x1a75b4b00ULL, 0x18db37e8aULL, 0x0f872e54cULL, 0x1c24afea4ULL,
    0x01e41e9fcULL, 0x04c144932ULL, 0x086d8e4d2ULL, 0x0271d9844ULL,
    0x160f7af7aULL, 0x052148f02ULL, 0x05bb8f1bcULL, 0x08e766a0cULL,
    0x0a90fd27aULL, 0x0a3c6f37aULL, 0x0b3af077aULL, 0x093a5f730ULL,
    0x04984d782ULL, 0x1d22c238eULL, 0x0ca6ef3acULL, 0x06cb08e5cULL,
    0x0234e0b26ULL, 0x063ded06aULL, 0x1d88abd4aULL, 0x06b749fb2ULL,
    0x04597456aULL, 0x04d56973cULL, 0x0e9e28eb4ULL, 0x1167f94f2ULL,
    0x07b3ff57aULL, 0x19385bf2eULL, 0x0c9c8b782ULL, 0x0cec3662eULL,
    0x13a9cba9eULL, 0x0e417f38aULL, 0x093e106a4ULL, 0x19329634aULL,
    0x167001a9cULL, 0x14e727980ULL, 0x1ddffc5d4ULL, 0x0e6fc4e6aULL,
    0x00df04680ULL, 0x0d104b8fcULL, 0x02342001eULL, 0x08227bb8aULL,
    0x00a2a8d7eULL, 0x05b397730ULL, 0x168763fa6ULL, 0x0b0cd4768ULL,
    0x1ed5a407aULL, 0x0e78eb416ULL, 0x0d2c3ed1aULL, 0x13c2b89c4ULL,
    0x0995a5724ULL, 0x1641378f0ULL, 0x19b1afbc4ULL, 0x0d7a4825cULL,
    0x109ffedc0ULL, 0x08d96551cULL, 0x0f2271e60ULL, 0x10f5ff2baULL,
    0x00b0bf8caULL, 0x00bf80dd2ULL, 0x123888b7aULL, 0x00167d312ULL,
    0x1e888f7dcULL, 0x18dcddd1cULL, 0x002ee03b2ULL, 0x0f6076544ULL,
    0x183e8d8feULL, 0x06a45d2b2ULL, 0x133d7a042ULL, 0x026f6a60aULL,
    0x116b0f50cULL, 0x1dd3e10e8ULL, 0x05fabe670ULL, 0x1a2adb74eULL,
    0x130004488ULL, 0x0de87806cULL, 0x000bcf5f6ULL, 0x19d34af3aULL,
    0x18f0c7078ULL, 0x014338754ULL, 0x017f27698ULL, 0x049c3cc9cULL,
    0x058ca5f00ULL, 0x15e3e77eeULL, 0x1af900c24ULL, 0x068bce87aULL,
    0x0b5cfca28ULL, 0x0dd07448eULL, 0x0ded288f8ULL, 0x1524fa6c6ULL,
    0x059f229bcULL, 0x1d8048348ULL, 0x06d390decULL, 0x16cba8acaULL,
    0x037170390ULL, 0x0a3e3e02cULL, 0x06353c1ccULL, 0x042d98888ULL,
    0x0c4584f5cULL, 0x0d73c7beaULL, 0x1f16a3418ULL, 0x1329d9f7eULL,
    0x0531377e2ULL, 0x185137662ULL, 0x1d8d9ca7cULL, 0x1b1c69528ULL,
    0x0b25b29f2ULL, 0x18a08b5bcULL, 0x19fb2a8b0ULL, 0x02178513aULL,
    0x1a08fe6acULL, 0x1da758ae0ULL, 0x045cddf4eULL, 0x0e0ac139eULL,
    0x1a91647f2ULL, 0x169cf9eb0ULL, 0x1a0f717c4ULL, 0x0170076faULL
};
// clang-format on

#if __SSE4_2__

uint32_t crc32c_hw_x86(uint32_t crc, const void * buf, size_t length)
{
    assert(data != nullptr);

    static const ssize_t kStepSize = sizeof(uint32_t);
    static const uint32_t kMaskOne = 0xFFFFFFFFUL;

    uint32_t crc32 = crc;

    const char * data = (const char *)buf;
    const char * data_end = (const char *)buf + length;
    ssize_t remain = length;

    do {
        if (likely(remain >= kStepSize)) {
            assert(data < data_end);
            uint32_t data32 = *(uint32_t *)(data);
            crc32 = _mm_crc32_u32(crc32, data32);
            data += kStepSize;
            remain -= kStepSize;
        }
        else {
            assert((data_end - data) >= 0 && (data_end - data) < kStepSize);
            assert((data_end - data) == remain);
            assert(remain >= 0);
            if (likely(remain > 0)) {
                uint32_t data32 = *(uint32_t *)(data);
                uint32_t rest = (uint32_t)(kStepSize - remain);
                assert(rest > 0 && rest < (uint32_t)kStepSize);
                uint32_t mask = kMaskOne >> (rest * 8U);
                data32 &= mask;
                crc32 = _mm_crc32_u32(crc32, data32);
            }
            break;
        }
    } while (1);

    return crc32;
}

uint32_t crc32c_hw_x64(uint32_t crc, const void * buf, size_t length)
{
#if CRC32_IS_X86_64
    assert(data != nullptr);

    static const ssize_t kStepSize = sizeof(uint64_t);
    static const uint64_t kMaskOne = 0xFFFFFFFFFFFFFFFFULL;

    uint64_t crc64 = crc;

    const char * data = (const char *)buf;
    const char * data_end = (const char *)buf + length;
    ssize_t remain = length;

    do {
        if (likely(remain >= kStepSize)) {
            assert(data < data_end);
            uint64_t data64 = *(uint64_t *)(data);
            crc64 = _mm_crc32_u64(crc64, data64);
            data += kStepSize;
            remain -= kStepSize;
        }
        else {
            assert((data_end - data) >= 0 && (data_end - data) < kStepSize);
            assert((data_end - data) == remain);
            assert(remain >= 0);
            if (likely(remain > 0)) {
                uint64_t data64 = *(uint64_t *)(data);
                size_t rest = (size_t)(kStepSize - remain);
                assert(rest > 0 && rest < (size_t)kStepSize);
                uint64_t mask = kMaskOne >> (rest * 8U);
                data64 &= mask;
                crc64 = _mm_crc32_u64(crc64, data64);
            }
            break;
        }
    } while (1);

    return (uint32_t)crc64;
#else
    return crc32c_hw_x86(crc, buf, length);
#endif // CRC32_IS_X86_64
}

uint32_t crc32c_hw_u32(uint32_t crc, const void * buf, size_t length)
{
    assert(data != nullptr);
    uint32_t crc32 = crc;

    static const size_t kStepSize = sizeof(uint32_t);
    const char * data = (const char *)buf;
    uint32_t * src = (uint32_t *)data;
    uint32_t * src_end = src + (length / kStepSize);

    while (likely(src < src_end)) {
        crc32 = _mm_crc32_u32(crc32, *src);
        ++src;
    }

    unsigned char * src8 = (unsigned char *)src;
    unsigned char * src8_end = (unsigned char *)(data + length);

    while (likely(src8 < src8_end)) {
        crc32 = _mm_crc32_u8(crc32, *src8);
        ++src8;
    }
    return crc32;
}

uint32_t crc32c_hw_u64(uint32_t crc, const void * buf, size_t length)
{
#if CRC32_IS_X86_64
    assert(data != nullptr);
    uint64_t crc64 = crc;

    static const size_t kStepSize = sizeof(uint64_t);
    const char * data = (const char *)buf;
    uint64_t * src = (uint64_t *)data;
    uint64_t * src_end = src + (length / kStepSize);

    while (likely(src < src_end)) {
        crc64 = _mm_crc32_u64(crc64, *src);
        ++src;
    }

    uint32_t crc32 = (uint32_t)crc64;
    unsigned char * src8 = (unsigned char *)src;
    unsigned char * src8_end = (unsigned char *)(data + length);

    while (likely(src8 < src8_end)) {
        crc32 = _mm_crc32_u8(crc32, *src8);
        ++src8;
    }
    return crc32;
#else
    return crc32c_hw_u32(crc, buf, length);
#endif // CRC32_IS_X86_64
}

static inline uint64_t crc32c_combine_crc_u32(size_t block_size, uint32_t crc0, uint32_t crc1, uint32_t crc2, const uint64_t * next2) {
    assert(block_size > 0 && block_size <= (sizeof(crc32c_clmul_constants) / 2));
    const __m128i multiplier = _mm_loadu_si128(reinterpret_cast<const __m128i *>(crc32c_clmul_constants) + block_size - 1);
    const __m128i crc0_xmm = _mm_cvtsi32_si128((int32_t)crc0);
    const __m128i result0  = _mm_clmulepi64_si128(crc0_xmm, multiplier, 0x00);
    const __m128i crc1_xmm = _mm_cvtsi32_si128((int32_t)crc1);
    const __m128i result1  = _mm_clmulepi64_si128(crc1_xmm, multiplier, 0x10);
    const __m128i result   = _mm_xor_si128(result0, result1);
    const __m128i __next2  = _mm_loadu_si128(reinterpret_cast<const __m128i *>((uint64_t *)next2 - 2));
    const __m128i result64 = _mm_xor_si128(result, __next2);
    uint32_t crc0_low  = _mm_cvtsi128_si32(result64);
    uint32_t crc32     = _mm_crc32_u32(crc2, crc0_low);
    uint32_t crc0_high = _mm_extract_epi32(result64, 0x01);
    crc32              = _mm_crc32_u32(crc32, crc0_high);    
    return crc32;
}

#if CRC32C_IS_X86_64

/*
 * crc32c_combine_crc() performs pclmulqdq multiplication of 2 partial CRC's and a well
 * chosen constant and xor's these with the remaining CRC.
 */
static inline uint64_t crc32c_combine_crc_u64(size_t block_size, uint64_t crc0, uint64_t crc1, uint64_t crc2, const uint64_t * next2) {
    assert(block_size > 0 && block_size <= (sizeof(crc32c_clmul_constants) / 2));
    const __m128i multiplier = _mm_loadu_si128(reinterpret_cast<const __m128i *>(crc32c_clmul_constants) + block_size - 1);
    const __m128i crc0_xmm = _mm_cvtsi64_si128((int64_t)crc0);
    const __m128i result0  = _mm_clmulepi64_si128(crc0_xmm, multiplier, 0x00);
    const __m128i crc1_xmm = _mm_cvtsi64_si128((int64_t)crc1);
    const __m128i result1  = _mm_clmulepi64_si128(crc1_xmm, multiplier, 0x10);
    const __m128i result   = _mm_xor_si128(result0, result1);
    crc0 = (uint64_t)_mm_cvtsi128_si64(result);
    crc0 = crc0 ^ *((uint64_t *)next2 - 1);
    uint64_t crc32 = _mm_crc32_u64(crc2, crc0);
    return crc32;
}

#endif // CRC32C_IS_X86_64

static inline uint32_t __crc32c_hw_u32(const char * data, size_t length, uint32_t crc_init)
{
    assert(data != nullptr);
    uint32_t crc32 = crc_init;

    static const size_t kStepSize = sizeof(uint32_t);
    uint32_t * src = (uint32_t *)data;
    uint32_t * src_end = src + (length / kStepSize);

    while (likely(src < src_end)) {
        crc32 = _mm_crc32_u32(crc32, *src);
        ++src;
    }

    unsigned char * src8 = (unsigned char *)src;
    unsigned char * src8_end = (unsigned char *)(data + length);

    while (likely(src8 < src8_end)) {
        crc32 = _mm_crc32_u8(crc32, *src8);
        ++src8;
    }
    return crc32;
}

#if CRC32C_IS_X86_64

static inline uint32_t __crc32c_hw_u64(const char * data, size_t length, uint32_t crc_init)
{
    static const size_t kStepSize = sizeof(uint64_t);
    static const size_t kAlignment = sizeof(uint64_t);
    static const size_t kLoopSize = 3 * kStepSize;

    assert(data != nullptr);
    uint32_t crc32 = crc_init;

    unsigned char * src8 = (unsigned char *)data;
    size_t data_len = length;

    size_t unaligned = ((kAlignment - (size_t)src8) & (kAlignment - 1));
    if (likely(unaligned != 0)) {
        length -= unaligned;
        if (likely(unaligned & 0x04U)) {
            crc32 = _mm_crc32_u32(crc32, *(uint32_t *)src8);
            src8 += sizeof(uint32_t);
        }
        if (likely(unaligned & 0x02U)) {
            crc32 = _mm_crc32_u16(crc32, *(uint16_t *)src8);
            src8 += sizeof(uint16_t);
        }
        if (likely(unaligned & 0x01U)) {
            crc32 = _mm_crc32_u8(crc32, *(uint8_t *)src8);
            src8 += sizeof(uint8_t);
        }
    }

    uint64_t crc64;
    uint64_t * src = (uint64_t *)src8;

    if (likely(length >= kLoopSize * 4)) {
        static const size_t kMaxBlockSize = 128;

        uint64_t crc0 = (uint64_t)crc32;
        uint64_t crc1 = 0;
        uint64_t crc2 = 0;

        size_t block_size;
        size_t loops = length / kLoopSize;
        length = length % kLoopSize;
        
        while (likely(loops > 0)) {
            block_size = (likely(loops >= kMaxBlockSize)) ? kMaxBlockSize : loops;
            assert(block_size >= 1);

            uint64_t * next0 = src;
            uint64_t * next1 = src + 1 * block_size;
            uint64_t * next2 = src + 2 * block_size;
#if 0
            size_t loop = block_size / 2;
            do {
                crc0 = _mm_crc32_u64(crc0, *next0);
                crc1 = _mm_crc32_u64(crc1, *next1);
                crc2 = _mm_crc32_u64(crc2, *next2);
                ++next0;
                ++next1;
                ++next2;
                --loop;
                crc0 = _mm_crc32_u64(crc0, *next0);
                crc1 = _mm_crc32_u64(crc1, *next1);
                if (likely(loop > 0))
                    crc2 = _mm_crc32_u64(crc2, *next2);
                ++next0;
                ++next1;
                ++next2;
            } while (likely(loop > 0));
#else
            size_t loop = block_size - 1;
            while (likely(loop > 0)) {
                crc0 = _mm_crc32_u64(crc0, *next0);
                crc1 = _mm_crc32_u64(crc1, *next1);
                crc2 = _mm_crc32_u64(crc2, *next2);
                ++next0;
                ++next1;
                ++next2;
                --loop;
            }

            crc0 = _mm_crc32_u64(crc0, *next0);
            crc1 = _mm_crc32_u64(crc1, *next1);
            ++next0;
            ++next1;
            ++next2;
#endif
            crc0 = crc32c_combine_crc_u64(block_size, crc0, crc1, crc2, next2);
            crc1 = crc2 = 0;

            src = next2;
            loops -= block_size;
            //length -= (kLoopSize * block_size);
        }

        crc64 = crc0;
    }
    else {
        // Convent crc32 to 64 bit integer.
        crc64 = (uint64_t)crc32;
    }

    uint64_t * src_end = src + (length / kStepSize);

    while (likely(src < src_end)) {
        crc64 = _mm_crc32_u64(crc64, *src);
        ++src;
    }

    // Pack the crc64 to 32 bit integer.
    crc32 = (uint32_t)crc64;

    unsigned char * src8_end;
    src8 = (unsigned char *)src;
    src8_end = (unsigned char *)(data + data_len);
    assert(src8 <= src8_end);

    size_t remain = (size_t)(src8_end - src8);
    assert(remain >= 0 && remain < kStepSize);
    if (likely(remain != 0)) {
        if (likely(remain & 0x04U)) {
            crc32 = _mm_crc32_u32(crc32, *(uint32_t *)src8);
            src8 += sizeof(uint32_t);
        }
        if (likely(remain & 0x02U)) {
            crc32 = _mm_crc32_u16(crc32, *(uint16_t *)src8);
            src8 += sizeof(uint16_t);
        }
        if (likely(remain & 0x01U)) {
            crc32 = _mm_crc32_u8(crc32, *(uint8_t *)src8);
            src8 += sizeof(uint32_t);
        }
    }
    assert(src8 == src8_end);
    return crc32;
}

#endif // CRC32C_IS_X86_64

uint32_t crc32c_hw(uint32_t crc_init, const void * data, size_t length)
{
#if CRC32C_IS_X86_64
    return __crc32c_hw_u64((const char *)data, length, crc_init);
#else
    return __crc32c_hw_u32((const char *)data, length, crc_init);
#endif
}

#endif // __SSE4_2__

} // namespace logging

#ifdef ssize_t
#undef ssize_t
#endif

// kate: indent-mode cstyle; indent-width 4; replace-tabs on; 
