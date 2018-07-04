
#include "logging/crc32c.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>

#ifdef __SSE4_2__
#ifdef _MSC_VER
#include <nmmintrin.h>  // For SSE 4.2
#else
#include <x86intrin.h>
//#include <nmmintrin.h>  // For SSE 4.2
#endif // _MSC_VER
#endif // __SSE4_2__

#ifndef ssize_t
#define ssize_t ptrdiff_t
#endif

namespace logging {

#if __SSE4_2__

uint32_t crc32c_hw_x86(uint32_t crc, const void * buf, size_t length)
{
    assert(data != nullptr);

    static const ssize_t kStepLen = sizeof(uint32_t);
    static const uint32_t kOneMask = 0xFFFFFFFFUL;

    uint32_t crc32 = crc;

    uint32_t data32;
    const char * data = (const char *)buf;
    const char * data_end = data + length;
    ssize_t remain = length;

    do {
        if (likely(remain >= kStepLen)) {
            assert(data < data_end);
            data32 = *(uint32_t *)(data);
            crc32 = _mm_crc32_u32(crc32, data32);
            data += kStepLen;
            remain -= kStepLen;
        }
        else {
            assert((data_end - data) >= 0 && (data_end - data) < kStepLen);
            assert((data_end - data) == remain);
            assert(remain >= 0);
            if (likely(remain > 0)) {
                data32 = *(uint32_t *)(data);
                uint32_t rest = (uint32_t)(kStepLen - remain);
                assert(rest > 0 && rest < (uint32_t)kStepLen);
                uint32_t mask = kOneMask >> (rest * 8U);
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

    static const ssize_t kStepLen = sizeof(uint64_t);
    static const uint64_t kOneMask = 0xFFFFFFFFFFFFFFFFULL;

    uint64_t crc64 = crc;

    uint64_t data64;
    const char * data = (const char *)buf;
    const char * data_end = data + length;
    ssize_t remain = length;

    do {
        if (likely(remain >= kStepLen)) {
            assert(data < data_end);
            data64 = *(uint64_t *)(data);
            crc64 = _mm_crc32_u64(crc64, data64);
            data += kStepLen;
            remain -= kStepLen;
        }
        else {
            assert((data_end - data) >= 0 && (data_end - data) < kStepLen);
            assert((data_end - data) == remain);
            assert(remain >= 0);
            if (likely(remain > 0)) {
                data64 = *(uint64_t *)(data);
                size_t rest = (size_t)(kStepLen - remain);
                assert(rest > 0 && rest < (size_t)kStepLen);
                uint64_t mask = kOneMask >> (rest * 8U);
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

    static const size_t kStepLen = sizeof(uint32_t);
    const char * data = (const char *)buf;
    uint32_t * src = (uint32_t *)data;
    uint32_t * src_end = src + (length / kStepLen);

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

    static const size_t kStepLen = sizeof(uint64_t);
    const char * data = (const char *)buf;
    uint64_t * src = (uint64_t *)data;
    uint64_t * src_end = src + (length / kStepLen);

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

#endif // __SSE4_2__

} // namespace logging

#ifdef ssize_t
#undef ssize_t
#endif
