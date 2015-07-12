#ifndef MIDNIGHT_CL
#define MIDNIGHT_CL

#if __ENDIAN_LITTLE__
#define SPH_LITTLE_ENDIAN 1
#else
#define SPH_BIG_ENDIAN 1
#endif

#define SPH_UPTR sph_u64

typedef unsigned int sph_u32;
typedef int sph_s32;
#ifndef __OPENCL_VERSION__
typedef unsigned long long sph_u64;
typedef long long sph_s64;
#else
typedef unsigned long sph_u64;
typedef long sph_s64;
#endif

#define SPH_64 1
#define SPH_64_TRUE 1

#define SPH_C32(x)    ((sph_u32)(x ## U))
#define SPH_T32(x) (as_uint(x))
#define SPH_ROTL32(x, n) rotate(as_uint(x), as_uint(n))
#define SPH_ROTR32(x, n)   SPH_ROTL32(x, (32 - (n)))

#define SPH_C64(x)    ((sph_u64)(x ## UL))
#define SPH_T64(x) (as_ulong(x))
#define SPH_ROTL64(x, n) rotate(as_ulong(x), (n) & 0xFFFFFFFFFFFFFFFFUL)
#define SPH_ROTR64(x, n)   SPH_ROTL64(x, (64 - (n)))

#include "bmw.cl"

#define SWAP4(x) as_uint(as_uchar4(x).wzyx)
#define SWAP8(x) as_ulong(as_uchar8(x).s76543210)

#define SHL(x, n)            ((x) << (n))
#define SHR(x, n)            ((x) >> (n))

#define CONST_EXP2  q[i+0] + SPH_ROTL64(q[i+1], 5)  + q[i+2] + SPH_ROTL64(q[i+3], 11) + \
                    q[i+4] + SPH_ROTL64(q[i+5], 27) + q[i+6] + SPH_ROTL64(q[i+7], 32) + \
                    q[i+8] + SPH_ROTL64(q[i+9], 37) + q[i+10] + SPH_ROTL64(q[i+11], 43) + \
                    q[i+12] + SPH_ROTL64(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])

#if SPH_BIG_ENDIAN
  #define DEC64E(x) (x)
  #define DEC64BE(x) (*(const __global sph_u64 *) (x));
#else
  #define DEC64E(x) SWAP8(x)
  #define DEC64BE(x) SWAP8(*(const __global sph_u64 *) (x));
#endif

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(__global unsigned char* block, volatile __global uint* output, const ulong target)
{
  uint gid = get_global_id(0);
  union {
    unsigned char h1[64];
    uint h4[16];
    ulong h8[8];
  } hash;

  
  // bmw
  sph_u64 BMW_H[16];
#pragma unroll 16
  for(unsigned u = 0; u < 16; u++)
    BMW_H[u] = BMW_IV512[u];

  sph_u64 mv[16],q[32];
  sph_u64 tmp;

  mv[ 0] = SWAP8(hash.h8[0]);
  mv[ 1] = SWAP8(hash.h8[1]);
  mv[ 2] = SWAP8(hash.h8[2]);
  mv[ 3] = SWAP8(hash.h8[3]);
  mv[ 4] = SWAP8(hash.h8[4]);
  mv[ 5] = SWAP8(hash.h8[5]);
  mv[ 6] = SWAP8(hash.h8[6]);
  mv[ 7] = SWAP8(hash.h8[7]);
  mv[ 8] = 0x80;
  mv[ 9] = 0;
  mv[10] = 0;
  mv[11] = 0;
  mv[12] = 0;
  mv[13] = 0;
  mv[14] = 0;
  mv[15] = SPH_C64(512);

  tmp = (mv[ 5] ^ BMW_H[ 5]) - (mv[ 7] ^ BMW_H[ 7]) + (mv[10] ^ BMW_H[10]) + (mv[13] ^ BMW_H[13]) + (mv[14] ^ BMW_H[14]);
  q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ SPH_ROTL64(tmp,  4) ^ SPH_ROTL64(tmp, 37)) + BMW_H[1];
  tmp = (mv[ 6] ^ BMW_H[ 6]) - (mv[ 8] ^ BMW_H[ 8]) + (mv[11] ^ BMW_H[11]) + (mv[14] ^ BMW_H[14]) - (mv[15] ^ BMW_H[15]);
  q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 13) ^ SPH_ROTL64(tmp, 43)) + BMW_H[2];
  tmp = (mv[ 0] ^ BMW_H[ 0]) + (mv[ 7] ^ BMW_H[ 7]) + (mv[ 9] ^ BMW_H[ 9]) - (mv[12] ^ BMW_H[12]) + (mv[15] ^ BMW_H[15]);
  q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ SPH_ROTL64(tmp, 19) ^ SPH_ROTL64(tmp, 53)) + BMW_H[3];
  tmp = (mv[ 0] ^ BMW_H[ 0]) - (mv[ 1] ^ BMW_H[ 1]) + (mv[ 8] ^ BMW_H[ 8]) - (mv[10] ^ BMW_H[10]) + (mv[13] ^ BMW_H[13]);
  q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 28) ^ SPH_ROTL64(tmp, 59)) + BMW_H[4];
  tmp = (mv[ 1] ^ BMW_H[ 1]) + (mv[ 2] ^ BMW_H[ 2]) + (mv[ 9] ^ BMW_H[ 9]) - (mv[11] ^ BMW_H[11]) - (mv[14] ^ BMW_H[14]);
  q[4] = (SHR(tmp, 1) ^ tmp) + BMW_H[5];
  tmp = (mv[ 3] ^ BMW_H[ 3]) - (mv[ 2] ^ BMW_H[ 2]) + (mv[10] ^ BMW_H[10]) - (mv[12] ^ BMW_H[12]) + (mv[15] ^ BMW_H[15]);
  q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ SPH_ROTL64(tmp,  4) ^ SPH_ROTL64(tmp, 37)) + BMW_H[6];
  tmp = (mv[ 4] ^ BMW_H[ 4]) - (mv[ 0] ^ BMW_H[ 0]) - (mv[ 3] ^ BMW_H[ 3]) - (mv[11] ^ BMW_H[11]) + (mv[13] ^ BMW_H[13]);
  q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 13) ^ SPH_ROTL64(tmp, 43)) + BMW_H[7];
  tmp = (mv[ 1] ^ BMW_H[ 1]) - (mv[ 4] ^ BMW_H[ 4]) - (mv[ 5] ^ BMW_H[ 5]) - (mv[12] ^ BMW_H[12]) - (mv[14] ^ BMW_H[14]);
  q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ SPH_ROTL64(tmp, 19) ^ SPH_ROTL64(tmp, 53)) + BMW_H[8];
  tmp = (mv[ 2] ^ BMW_H[ 2]) - (mv[ 5] ^ BMW_H[ 5]) - (mv[ 6] ^ BMW_H[ 6]) + (mv[13] ^ BMW_H[13]) - (mv[15] ^ BMW_H[15]);
  q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 28) ^ SPH_ROTL64(tmp, 59)) + BMW_H[9];
  tmp = (mv[ 0] ^ BMW_H[ 0]) - (mv[ 3] ^ BMW_H[ 3]) + (mv[ 6] ^ BMW_H[ 6]) - (mv[ 7] ^ BMW_H[ 7]) + (mv[14] ^ BMW_H[14]);
  q[9] = (SHR(tmp, 1) ^ tmp) + BMW_H[10];
  tmp = (mv[ 8] ^ BMW_H[ 8]) - (mv[ 1] ^ BMW_H[ 1]) - (mv[ 4] ^ BMW_H[ 4]) - (mv[ 7] ^ BMW_H[ 7]) + (mv[15] ^ BMW_H[15]);
  q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ SPH_ROTL64(tmp,  4) ^ SPH_ROTL64(tmp, 37)) + BMW_H[11];
  tmp = (mv[ 8] ^ BMW_H[ 8]) - (mv[ 0] ^ BMW_H[ 0]) - (mv[ 2] ^ BMW_H[ 2]) - (mv[ 5] ^ BMW_H[ 5]) + (mv[ 9] ^ BMW_H[ 9]);
  q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 13) ^ SPH_ROTL64(tmp, 43)) + BMW_H[12];
  tmp = (mv[ 1] ^ BMW_H[ 1]) + (mv[ 3] ^ BMW_H[ 3]) - (mv[ 6] ^ BMW_H[ 6]) - (mv[ 9] ^ BMW_H[ 9]) + (mv[10] ^ BMW_H[10]);
  q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ SPH_ROTL64(tmp, 19) ^ SPH_ROTL64(tmp, 53)) + BMW_H[13];
  tmp = (mv[ 2] ^ BMW_H[ 2]) + (mv[ 4] ^ BMW_H[ 4]) + (mv[ 7] ^ BMW_H[ 7]) + (mv[10] ^ BMW_H[10]) + (mv[11] ^ BMW_H[11]);
  q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 28) ^ SPH_ROTL64(tmp, 59)) + BMW_H[14];
  tmp = (mv[ 3] ^ BMW_H[ 3]) - (mv[ 5] ^ BMW_H[ 5]) + (mv[ 8] ^ BMW_H[ 8]) - (mv[11] ^ BMW_H[11]) - (mv[12] ^ BMW_H[12]);
  q[14] = (SHR(tmp, 1) ^ tmp) + BMW_H[15];
  tmp = (mv[12] ^ BMW_H[12]) - (mv[ 4] ^ BMW_H[ 4]) - (mv[ 6] ^ BMW_H[ 6]) - (mv[ 9] ^ BMW_H[ 9]) + (mv[13] ^ BMW_H[13]);
  q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ SPH_ROTL64(tmp, 4) ^ SPH_ROTL64(tmp, 37)) + BMW_H[0];

#pragma unroll 2
  for(int i=0;i<2;i++)
  {
    q[i+16] =
    (SHR(q[i], 1) ^ SHL(q[i], 2) ^ SPH_ROTL64(q[i], 13) ^ SPH_ROTL64(q[i], 43)) +
    (SHR(q[i+1], 2) ^ SHL(q[i+1], 1) ^ SPH_ROTL64(q[i+1], 19) ^ SPH_ROTL64(q[i+1], 53)) +
    (SHR(q[i+2], 2) ^ SHL(q[i+2], 2) ^ SPH_ROTL64(q[i+2], 28) ^ SPH_ROTL64(q[i+2], 59)) +
    (SHR(q[i+3], 1) ^ SHL(q[i+3], 3) ^ SPH_ROTL64(q[i+3],  4) ^ SPH_ROTL64(q[i+3], 37)) +
    (SHR(q[i+4], 1) ^ SHL(q[i+4], 2) ^ SPH_ROTL64(q[i+4], 13) ^ SPH_ROTL64(q[i+4], 43)) +
    (SHR(q[i+5], 2) ^ SHL(q[i+5], 1) ^ SPH_ROTL64(q[i+5], 19) ^ SPH_ROTL64(q[i+5], 53)) +
    (SHR(q[i+6], 2) ^ SHL(q[i+6], 2) ^ SPH_ROTL64(q[i+6], 28) ^ SPH_ROTL64(q[i+6], 59)) +
    (SHR(q[i+7], 1) ^ SHL(q[i+7], 3) ^ SPH_ROTL64(q[i+7],  4) ^ SPH_ROTL64(q[i+7], 37)) +
    (SHR(q[i+8], 1) ^ SHL(q[i+8], 2) ^ SPH_ROTL64(q[i+8], 13) ^ SPH_ROTL64(q[i+8], 43)) +
    (SHR(q[i+9], 2) ^ SHL(q[i+9], 1) ^ SPH_ROTL64(q[i+9], 19) ^ SPH_ROTL64(q[i+9], 53)) +
    (SHR(q[i+10], 2) ^ SHL(q[i+10], 2) ^ SPH_ROTL64(q[i+10], 28) ^ SPH_ROTL64(q[i+10], 59)) +
    (SHR(q[i+11], 1) ^ SHL(q[i+11], 3) ^ SPH_ROTL64(q[i+11],  4) ^ SPH_ROTL64(q[i+11], 37)) +
    (SHR(q[i+12], 1) ^ SHL(q[i+12], 2) ^ SPH_ROTL64(q[i+12], 13) ^ SPH_ROTL64(q[i+12], 43)) +
    (SHR(q[i+13], 2) ^ SHL(q[i+13], 1) ^ SPH_ROTL64(q[i+13], 19) ^ SPH_ROTL64(q[i+13], 53)) +
    (SHR(q[i+14], 2) ^ SHL(q[i+14], 2) ^ SPH_ROTL64(q[i+14], 28) ^ SPH_ROTL64(q[i+14], 59)) +
    (SHR(q[i+15], 1) ^ SHL(q[i+15], 3) ^ SPH_ROTL64(q[i+15],  4) ^ SPH_ROTL64(q[i+15], 37)) +
    ((  ((i+16)*(0x0555555555555555ul)) + SPH_ROTL64(mv[i], i+1) +
      SPH_ROTL64(mv[i+3], i+4) - SPH_ROTL64(mv[i+10], i+11) ) ^ BMW_H[i+7]);
  }

#pragma unroll 4
  for(int i=2;i<6;i++) {
    q[i+16] = CONST_EXP2 +
    ((  ((i+16)*(0x0555555555555555ul)) + SPH_ROTL64(mv[i], i+1) +
      SPH_ROTL64(mv[i+3], i+4) - SPH_ROTL64(mv[i+10], i+11) ) ^ BMW_H[i+7]);
  }
#pragma unroll 3
  for(int i=6;i<9;i++) {
    q[i+16] = CONST_EXP2 +
    ((  ((i+16)*(0x0555555555555555ul)) + SPH_ROTL64(mv[i], i+1) +
      SPH_ROTL64(mv[i+3], i+4) - SPH_ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i+7]);
  }
#pragma unroll 4
  for(int i=9;i<13;i++) {
    q[i+16] = CONST_EXP2 +
    ((  ((i+16)*(0x0555555555555555ul)) + SPH_ROTL64(mv[i], i+1) +
      SPH_ROTL64(mv[i+3], i+4) - SPH_ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i-9]);
  }
#pragma unroll 3
  for(int i=13;i<16;i++) {
    q[i+16] = CONST_EXP2 +
    ((  ((i+16)*(0x0555555555555555ul)) + SPH_ROTL64(mv[i], i+1) +
      SPH_ROTL64(mv[i-13], (i-13)+1) - SPH_ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i-9]);
  }

sph_u64 XL64 = q[16]^q[17]^q[18]^q[19]^q[20]^q[21]^q[22]^q[23];
sph_u64 XH64 =  XL64^q[24]^q[25]^q[26]^q[27]^q[28]^q[29]^q[30]^q[31];

  BMW_H[0] =             (SHL(XH64, 5) ^ SHR(q[16],5) ^ mv[ 0]) + (  XL64  ^ q[24] ^ q[ 0]);
  BMW_H[1] =             (SHR(XH64, 7) ^ SHL(q[17],8) ^ mv[ 1]) + (  XL64  ^ q[25] ^ q[ 1]);
  BMW_H[2] =             (SHR(XH64, 5) ^ SHL(q[18],5) ^ mv[ 2]) + (  XL64  ^ q[26] ^ q[ 2]);
  BMW_H[3] =             (SHR(XH64, 1) ^ SHL(q[19],5) ^ mv[ 3]) + (  XL64  ^ q[27] ^ q[ 3]);
  BMW_H[4] =             (SHR(XH64, 3) ^     q[20]    ^ mv[ 4]) + (  XL64  ^ q[28] ^ q[ 4]);
  BMW_H[5] =             (SHL(XH64, 6) ^ SHR(q[21],6) ^ mv[ 5]) + (  XL64  ^ q[29] ^ q[ 5]);
  BMW_H[6] =             (SHR(XH64, 4) ^ SHL(q[22],6) ^ mv[ 6]) + (  XL64  ^ q[30] ^ q[ 6]);
  BMW_H[7] =             (SHR(XH64,11) ^ SHL(q[23],2) ^ mv[ 7]) + (  XL64  ^ q[31] ^ q[ 7]);

  BMW_H[ 8] = SPH_ROTL64(BMW_H[4], 9) + (  XH64   ^   q[24]  ^ mv[ 8]) + (SHL(XL64,8) ^ q[23] ^ q[ 8]);
  BMW_H[ 9] = SPH_ROTL64(BMW_H[5],10) + (  XH64   ^   q[25]  ^ mv[ 9]) + (SHR(XL64,6) ^ q[16] ^ q[ 9]);
  BMW_H[10] = SPH_ROTL64(BMW_H[6],11) + (  XH64   ^   q[26]  ^ mv[10]) + (SHL(XL64,6) ^ q[17] ^ q[10]);
  BMW_H[11] = SPH_ROTL64(BMW_H[7],12) + (  XH64   ^   q[27]  ^ mv[11]) + (SHL(XL64,4) ^ q[18] ^ q[11]);
  BMW_H[12] = SPH_ROTL64(BMW_H[0],13) + (  XH64   ^   q[28]  ^ mv[12]) + (SHR(XL64,3) ^ q[19] ^ q[12]);
  BMW_H[13] = SPH_ROTL64(BMW_H[1],14) + (  XH64   ^   q[29]  ^ mv[13]) + (SHR(XL64,4) ^ q[20] ^ q[13]);
  BMW_H[14] = SPH_ROTL64(BMW_H[2],15) + (  XH64   ^   q[30]  ^ mv[14]) + (SHR(XL64,7) ^ q[21] ^ q[14]);
  BMW_H[15] = SPH_ROTL64(BMW_H[3],16) + (  XH64   ^   q[31]  ^ mv[15]) + (SHR(XL64,2) ^ q[22] ^ q[15]);

#pragma unroll 16
     for(int i=0;i<16;i++) {
       mv[i] = BMW_H[i];
       BMW_H[i] = 0xaaaaaaaaaaaaaaa0ul + (sph_u64)i;
     }

  tmp = (mv[ 5] ^ BMW_H[ 5]) - (mv[ 7] ^ BMW_H[ 7]) + (mv[10] ^ BMW_H[10]) + (mv[13] ^ BMW_H[13]) + (mv[14] ^ BMW_H[14]);
  q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ SPH_ROTL64(tmp,  4) ^ SPH_ROTL64(tmp, 37)) + BMW_H[1];
  tmp = (mv[ 6] ^ BMW_H[ 6]) - (mv[ 8] ^ BMW_H[ 8]) + (mv[11] ^ BMW_H[11]) + (mv[14] ^ BMW_H[14]) - (mv[15] ^ BMW_H[15]);
  q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 13) ^ SPH_ROTL64(tmp, 43)) + BMW_H[2];
  tmp = (mv[ 0] ^ BMW_H[ 0]) + (mv[ 7] ^ BMW_H[ 7]) + (mv[ 9] ^ BMW_H[ 9]) - (mv[12] ^ BMW_H[12]) + (mv[15] ^ BMW_H[15]);
  q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ SPH_ROTL64(tmp, 19) ^ SPH_ROTL64(tmp, 53)) + BMW_H[3];
  tmp = (mv[ 0] ^ BMW_H[ 0]) - (mv[ 1] ^ BMW_H[ 1]) + (mv[ 8] ^ BMW_H[ 8]) - (mv[10] ^ BMW_H[10]) + (mv[13] ^ BMW_H[13]);
  q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 28) ^ SPH_ROTL64(tmp, 59)) + BMW_H[4];
  tmp = (mv[ 1] ^ BMW_H[ 1]) + (mv[ 2] ^ BMW_H[ 2]) + (mv[ 9] ^ BMW_H[ 9]) - (mv[11] ^ BMW_H[11]) - (mv[14] ^ BMW_H[14]);
  q[4] = (SHR(tmp, 1) ^ tmp) + BMW_H[5];
  tmp = (mv[ 3] ^ BMW_H[ 3]) - (mv[ 2] ^ BMW_H[ 2]) + (mv[10] ^ BMW_H[10]) - (mv[12] ^ BMW_H[12]) + (mv[15] ^ BMW_H[15]);
  q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ SPH_ROTL64(tmp,  4) ^ SPH_ROTL64(tmp, 37)) + BMW_H[6];
  tmp = (mv[ 4] ^ BMW_H[ 4]) - (mv[ 0] ^ BMW_H[ 0]) - (mv[ 3] ^ BMW_H[ 3]) - (mv[11] ^ BMW_H[11]) + (mv[13] ^ BMW_H[13]);
  q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 13) ^ SPH_ROTL64(tmp, 43)) + BMW_H[7];
  tmp = (mv[ 1] ^ BMW_H[ 1]) - (mv[ 4] ^ BMW_H[ 4]) - (mv[ 5] ^ BMW_H[ 5]) - (mv[12] ^ BMW_H[12]) - (mv[14] ^ BMW_H[14]);
  q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ SPH_ROTL64(tmp, 19) ^ SPH_ROTL64(tmp, 53)) + BMW_H[8];
  tmp = (mv[ 2] ^ BMW_H[ 2]) - (mv[ 5] ^ BMW_H[ 5]) - (mv[ 6] ^ BMW_H[ 6]) + (mv[13] ^ BMW_H[13]) - (mv[15] ^ BMW_H[15]);
  q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 28) ^ SPH_ROTL64(tmp, 59)) + BMW_H[9];
  tmp = (mv[ 0] ^ BMW_H[ 0]) - (mv[ 3] ^ BMW_H[ 3]) + (mv[ 6] ^ BMW_H[ 6]) - (mv[ 7] ^ BMW_H[ 7]) + (mv[14] ^ BMW_H[14]);
  q[9] = (SHR(tmp, 1) ^ tmp) + BMW_H[10];
  tmp = (mv[ 8] ^ BMW_H[ 8]) - (mv[ 1] ^ BMW_H[ 1]) - (mv[ 4] ^ BMW_H[ 4]) - (mv[ 7] ^ BMW_H[ 7]) + (mv[15] ^ BMW_H[15]);
  q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ SPH_ROTL64(tmp,  4) ^ SPH_ROTL64(tmp, 37)) + BMW_H[11];
  tmp = (mv[ 8] ^ BMW_H[ 8]) - (mv[ 0] ^ BMW_H[ 0]) - (mv[ 2] ^ BMW_H[ 2]) - (mv[ 5] ^ BMW_H[ 5]) + (mv[ 9] ^ BMW_H[ 9]);
  q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 13) ^ SPH_ROTL64(tmp, 43)) + BMW_H[12];
  tmp = (mv[ 1] ^ BMW_H[ 1]) + (mv[ 3] ^ BMW_H[ 3]) - (mv[ 6] ^ BMW_H[ 6]) - (mv[ 9] ^ BMW_H[ 9]) + (mv[10] ^ BMW_H[10]);
  q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ SPH_ROTL64(tmp, 19) ^ SPH_ROTL64(tmp, 53)) + BMW_H[13];
  tmp = (mv[ 2] ^ BMW_H[ 2]) + (mv[ 4] ^ BMW_H[ 4]) + (mv[ 7] ^ BMW_H[ 7]) + (mv[10] ^ BMW_H[10]) + (mv[11] ^ BMW_H[11]);
  q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ SPH_ROTL64(tmp, 28) ^ SPH_ROTL64(tmp, 59)) + BMW_H[14];
  tmp = (mv[ 3] ^ BMW_H[ 3]) - (mv[ 5] ^ BMW_H[ 5]) + (mv[ 8] ^ BMW_H[ 8]) - (mv[11] ^ BMW_H[11]) - (mv[12] ^ BMW_H[12]);
  q[14] = (SHR(tmp, 1) ^ tmp) + BMW_H[15];
  tmp = (mv[12] ^ BMW_H[12]) - (mv[ 4] ^ BMW_H[ 4]) - (mv[ 6] ^ BMW_H[ 6]) - (mv[ 9] ^ BMW_H[ 9]) + (mv[13] ^ BMW_H[13]);
  q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ SPH_ROTL64(tmp, 4) ^ SPH_ROTL64(tmp, 37)) + BMW_H[0];


#pragma unroll 2
  for(int i=0;i<2;i++)
  {
    q[i+16] =
    (SHR(q[i], 1) ^ SHL(q[i], 2) ^ SPH_ROTL64(q[i], 13) ^ SPH_ROTL64(q[i], 43)) +
    (SHR(q[i+1], 2) ^ SHL(q[i+1], 1) ^ SPH_ROTL64(q[i+1], 19) ^ SPH_ROTL64(q[i+1], 53)) +
    (SHR(q[i+2], 2) ^ SHL(q[i+2], 2) ^ SPH_ROTL64(q[i+2], 28) ^ SPH_ROTL64(q[i+2], 59)) +
    (SHR(q[i+3], 1) ^ SHL(q[i+3], 3) ^ SPH_ROTL64(q[i+3],  4) ^ SPH_ROTL64(q[i+3], 37)) +
    (SHR(q[i+4], 1) ^ SHL(q[i+4], 2) ^ SPH_ROTL64(q[i+4], 13) ^ SPH_ROTL64(q[i+4], 43)) +
    (SHR(q[i+5], 2) ^ SHL(q[i+5], 1) ^ SPH_ROTL64(q[i+5], 19) ^ SPH_ROTL64(q[i+5], 53)) +
    (SHR(q[i+6], 2) ^ SHL(q[i+6], 2) ^ SPH_ROTL64(q[i+6], 28) ^ SPH_ROTL64(q[i+6], 59)) +
    (SHR(q[i+7], 1) ^ SHL(q[i+7], 3) ^ SPH_ROTL64(q[i+7],  4) ^ SPH_ROTL64(q[i+7], 37)) +
    (SHR(q[i+8], 1) ^ SHL(q[i+8], 2) ^ SPH_ROTL64(q[i+8], 13) ^ SPH_ROTL64(q[i+8], 43)) +
    (SHR(q[i+9], 2) ^ SHL(q[i+9], 1) ^ SPH_ROTL64(q[i+9], 19) ^ SPH_ROTL64(q[i+9], 53)) +
    (SHR(q[i+10], 2) ^ SHL(q[i+10], 2) ^ SPH_ROTL64(q[i+10], 28) ^ SPH_ROTL64(q[i+10], 59)) +
    (SHR(q[i+11], 1) ^ SHL(q[i+11], 3) ^ SPH_ROTL64(q[i+11],  4) ^ SPH_ROTL64(q[i+11], 37)) +
    (SHR(q[i+12], 1) ^ SHL(q[i+12], 2) ^ SPH_ROTL64(q[i+12], 13) ^ SPH_ROTL64(q[i+12], 43)) +
    (SHR(q[i+13], 2) ^ SHL(q[i+13], 1) ^ SPH_ROTL64(q[i+13], 19) ^ SPH_ROTL64(q[i+13], 53)) +
    (SHR(q[i+14], 2) ^ SHL(q[i+14], 2) ^ SPH_ROTL64(q[i+14], 28) ^ SPH_ROTL64(q[i+14], 59)) +
    (SHR(q[i+15], 1) ^ SHL(q[i+15], 3) ^ SPH_ROTL64(q[i+15],  4) ^ SPH_ROTL64(q[i+15], 37)) +
    ((  ((i+16)*(0x0555555555555555ul)) + SPH_ROTL64(mv[i], i+1) +
      SPH_ROTL64(mv[i+3], i+4) - SPH_ROTL64(mv[i+10], i+11) ) ^ BMW_H[i+7]);
  }

#pragma unroll 4
  for(int i=2;i<6;i++) {
    q[i+16] = CONST_EXP2 +
    ((  ((i+16)*(0x0555555555555555ul)) + SPH_ROTL64(mv[i], i+1) +
      SPH_ROTL64(mv[i+3], i+4) - SPH_ROTL64(mv[i+10], i+11) ) ^ BMW_H[i+7]);
  }
#pragma unroll 3
  for(int i=6;i<9;i++) {
    q[i+16] = CONST_EXP2 +
    ((  ((i+16)*(0x0555555555555555ul)) + SPH_ROTL64(mv[i], i+1) +
      SPH_ROTL64(mv[i+3], i+4) - SPH_ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i+7]);
  }
#pragma unroll 4
  for(int i=9;i<13;i++) {
    q[i+16] = CONST_EXP2 +
    ((  ((i+16)*(0x0555555555555555ul)) + SPH_ROTL64(mv[i], i+1) +
      SPH_ROTL64(mv[i+3], i+4) - SPH_ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i-9]);
  }
#pragma unroll 3
  for(int i=13;i<16;i++) {
    q[i+16] = CONST_EXP2 +
    ((  ((i+16)*(0x0555555555555555ul)) + SPH_ROTL64(mv[i], i+1) +
      SPH_ROTL64(mv[i-13], (i-13)+1) - SPH_ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i-9]);
  }

XL64 = q[16]^q[17]^q[18]^q[19]^q[20]^q[21]^q[22]^q[23];
XH64 =  XL64^q[24]^q[25]^q[26]^q[27]^q[28]^q[29]^q[30]^q[31];
  BMW_H[0] =             (SHL(XH64, 5) ^ SHR(q[16],5) ^ mv[ 0]) + (  XL64  ^ q[24] ^ q[ 0]);
  BMW_H[1] =             (SHR(XH64, 7) ^ SHL(q[17],8) ^ mv[ 1]) + (  XL64  ^ q[25] ^ q[ 1]);
  BMW_H[2] =             (SHR(XH64, 5) ^ SHL(q[18],5) ^ mv[ 2]) + (  XL64  ^ q[26] ^ q[ 2]);
  BMW_H[3] =             (SHR(XH64, 1) ^ SHL(q[19],5) ^ mv[ 3]) + (  XL64  ^ q[27] ^ q[ 3]);
  BMW_H[4] =             (SHR(XH64, 3) ^     q[20]    ^ mv[ 4]) + (  XL64  ^ q[28] ^ q[ 4]);
  BMW_H[5] =             (SHL(XH64, 6) ^ SHR(q[21],6) ^ mv[ 5]) + (  XL64  ^ q[29] ^ q[ 5]);
  BMW_H[6] =             (SHR(XH64, 4) ^ SHL(q[22],6) ^ mv[ 6]) + (  XL64  ^ q[30] ^ q[ 6]);
  BMW_H[7] =             (SHR(XH64,11) ^ SHL(q[23],2) ^ mv[ 7]) + (  XL64  ^ q[31] ^ q[ 7]);

  BMW_H[ 8] = SPH_ROTL64(BMW_H[4], 9) + (  XH64   ^   q[24]  ^ mv[ 8]) + (SHL(XL64,8) ^ q[23] ^ q[ 8]);
  BMW_H[ 9] = SPH_ROTL64(BMW_H[5],10) + (  XH64   ^   q[25]  ^ mv[ 9]) + (SHR(XL64,6) ^ q[16] ^ q[ 9]);
  BMW_H[10] = SPH_ROTL64(BMW_H[6],11) + (  XH64   ^   q[26]  ^ mv[10]) + (SHL(XL64,6) ^ q[17] ^ q[10]);
  BMW_H[11] = SPH_ROTL64(BMW_H[7],12) + (  XH64   ^   q[27]  ^ mv[11]) + (SHL(XL64,4) ^ q[18] ^ q[11]);
  BMW_H[12] = SPH_ROTL64(BMW_H[0],13) + (  XH64   ^   q[28]  ^ mv[12]) + (SHR(XL64,3) ^ q[19] ^ q[12]);
  BMW_H[13] = SPH_ROTL64(BMW_H[1],14) + (  XH64   ^   q[29]  ^ mv[13]) + (SHR(XL64,4) ^ q[20] ^ q[13]);
  BMW_H[14] = SPH_ROTL64(BMW_H[2],15) + (  XH64   ^   q[30]  ^ mv[14]) + (SHR(XL64,7) ^ q[21] ^ q[14]);
  BMW_H[15] = SPH_ROTL64(BMW_H[3],16) + (  XH64   ^   q[31]  ^ mv[15]) + (SHR(XL64,2) ^ q[22] ^ q[15]);

  hash.h8[0] = SWAP8(BMW_H[8]);
  hash.h8[1] = SWAP8(BMW_H[9]);
  hash.h8[2] = SWAP8(BMW_H[10]);
  hash.h8[3] = SWAP8(BMW_H[11]);
  hash.h8[4] = SWAP8(BMW_H[12]);
  hash.h8[5] = SWAP8(BMW_H[13]);
  hash.h8[6] = SWAP8(BMW_H[14]);
  hash.h8[7] = SWAP8(BMW_H[15]);

  

  bool result = (SWAP8(hash.h8[3]) <= target);
  if (result)
    output[output[0xFF]++] = SWAP4(gid);
}

#endif // MIDNIGHT_CL
