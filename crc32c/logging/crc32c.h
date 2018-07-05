// Copyright 2008,2009,2010 Massachusetts Institute of Technology.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#ifndef LOGGING_CRC32C_H__
#define LOGGING_CRC32C_H__

#include <cstddef>
#include <stdint.h>

#ifndef __SSE4_2__
#define __SSE4_2__  1
#endif

#if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
 || defined(__amd64__) || defined(__x86_64__)
#ifndef CRC32_IS_X86_64
#define CRC32_IS_X86_64     1
#endif
#endif // _WIN64 || __amd64__

#if (defined(__GNUC__) && ((__GNUC__ == 2 && __GNUC_MINOR__ >= 96) || (__GNUC__ >= 3))) \
 || (defined(__clang__) && ((__clang_major__ == 2 && __clang_minor__ >= 1) || (__clang_major__ >= 3)))
// Since gcc v2.96 or clang v2.1
#ifndef likely
#define likely(expr)        __builtin_expect(!!(expr), 1)
#endif
#ifndef unlikely
#define unlikely(expr)      __builtin_expect(!!(expr), 0)
#endif
#else // !likely() & unlikely()
#ifndef likely
#define likely(expr)        (expr)
#endif
#ifndef unlikely
#define unlikely(expr)      (expr)
#endif
#endif // likely() & unlikely()

namespace logging {

/** Returns the initial value for a CRC32-C computation. */
static inline uint32_t crc32cInit() {
    return 0xFFFFFFFF;
}

/** Pointer to a function that computes a CRC32C checksum.
@arg crc Previous CRC32C value, or crc32c_init().
@arg data Pointer to the data to be checksummed.
@arg length length of the data in bytes.
*/
typedef uint32_t (*CRC32CFunctionPtr)(uint32_t crc, const void* data, size_t length);

/** This will map automatically to the "best" CRC implementation. */
extern CRC32CFunctionPtr crc32c;

CRC32CFunctionPtr detectBestCRC32C();

/** Converts a partial CRC32-C computation to the final value. */
static inline uint32_t crc32cFinish(uint32_t crc) {
    return ~crc;
}

uint32_t crc32cSarwate(uint32_t crc, const void* data, size_t length);
uint32_t crc32cSlicingBy4(uint32_t crc, const void* data, size_t length);
uint32_t crc32cSlicingBy8(uint32_t crc, const void* data, size_t length);
uint32_t crc32cHardware32(uint32_t crc, const void* data, size_t length);
uint32_t crc32cHardware64(uint32_t crc, const void* data, size_t length);
uint32_t crc32cAdler(uint32_t crc, const void* data, size_t length);
uint32_t crc32cIntelC(uint32_t crc, const void* data, size_t length);
uint32_t crc32cIntelAsm(uint32_t crc, const void *buf, size_t len);

uint32_t crc32c_hw_x86(uint32_t crc, const void * data, size_t length);
uint32_t crc32c_hw_x64(uint32_t crc, const void * data, size_t length);
uint32_t crc32c_hw_u32(uint32_t crc, const void * data, size_t length);
uint32_t crc32c_hw_u64(uint32_t crc, const void * data, size_t length);
uint32_t crc32c_hw(uint32_t crc_init, const void * data, size_t length);

}  // namespace logging
#endif
