
#include "config.h"

#include <stdio.h>

#include "arith.h"


typedef struct _State State;
struct _State {
  int qe_value;
  int next_lps;
  int next_mps;
  int switch_mps;
};

static State states[] = {
  { 0x5a1d,  1,  1, 1 }, /* 0 */
  { 0x2586, 14,  2, 0 },
  { 0x1114, 16,  3, 0 },
  { 0x080b, 18,  4, 0 },
  { 0x03d8, 20,  5, 0 },
  { 0x01da, 23,  6, 0 }, /* 5 */
  { 0x00e5, 25,  7, 0 },
  { 0x006f, 28,  8, 0 },
  { 0x0036, 30,  9, 0 },
  { 0x001a, 33, 10, 0 },
  { 0x000d, 35, 11, 0 }, /* 10 */
  { 0x0006,  9, 12, 0 },
  { 0x0003, 10, 13, 0 },
  { 0x0001, 12, 13, 0 },
  { 0x5a7f, 15, 15, 1 },
  { 0x3f25, 36, 16, 0 }, /* 15 */
  { 0x2cf2, 38, 17, 0 },
  { 0x207c, 39, 18, 0 },
  { 0x17b9, 40, 18, 0 },
  { 0x1182, 42, 20, 0 },
  { 0x0cef, 43, 21, 0 }, /* 20 */
  { 0x09a1, 45, 22, 0 },
  { 0x072f, 46, 23, 0 },
  { 0x055c, 48, 24, 0 },
  { 0x0406, 49, 25, 0 },
  { 0x0303, 51, 26, 0 }, /* 25 */
  { 0x0240, 52, 27, 0 },
  { 0x01b1, 54, 28, 0 },
  { 0x0144, 56, 29, 0 },
  { 0x00f5, 57, 30, 0 },
  { 0x00b7, 59, 31, 0 }, /* 30 */
  { 0x008a, 60, 32, 0 },
  { 0x0068, 62, 33, 0 },
  { 0x004e, 63, 34, 0 },
  { 0x003b, 32, 35, 0 },
  { 0x002c, 33,  9, 0 }, /* 35 */
  { 0x5ae1, 37, 37, 1 },
  { 0x484c, 64, 38, 0 },
  { 0x3a0d, 65, 39, 0 },
  { 0x2ef1, 67, 40, 0 },
  { 0x261f, 68, 41, 0 }, /* 40 */
  { 0x1f33, 69, 42, 0 },
  { 0x1948, 70, 43, 0 },
  { 0x1518, 72, 44, 0 },
  { 0x1177, 73, 45, 0 },
  { 0x0e74, 74, 46, 0 }, /* 45 */
  { 0x0bfb, 75, 47, 0 },
  { 0x09f8, 77, 48, 0 },
  { 0x0861, 78, 49, 0 },
  { 0x0706, 79, 50, 0 },
  { 0x05cd, 48, 51, 0 }, /* 50 */
  { 0x04de, 50, 52, 0 },
  { 0x040f, 50, 53, 0 },
  { 0x0363, 51, 54, 0 },
  { 0x02d4, 52, 55, 0 },
  { 0x025c, 53, 56, 0 }, /* 55 */
  { 0x01f8, 54, 57, 0 },
  { 0x01a4, 55, 58, 0 },
  { 0x0160, 56, 59, 0 },
  { 0x0125, 57, 60, 0 },
  { 0x00f6, 58, 61, 0 }, /* 60 */
  { 0x00cb, 59, 62, 0 },
  { 0x00ab, 61, 63, 0 },
  { 0x008f, 61, 32, 0 },
  { 0x5b12, 65, 65, 1 },
  { 0x4d04, 80, 66, 0 }, /* 65 */
  { 0x412c, 81, 67, 0 },
  { 0x37d8, 82, 68, 0 },
  { 0x2fe8, 83, 69, 0 },
  { 0x293c, 84, 70, 0 },
  { 0x2379, 86, 71, 0 }, /* 70 */
  { 0x1edf, 87, 72, 0 },
  { 0x1aa9, 87, 73, 0 },
  { 0x174e, 72, 74, 0 },
  { 0x1424, 72, 75, 0 },
  { 0x119c, 74, 76, 0 }, /* 75 */
  { 0x0f6b, 74, 77, 0 },
  { 0x0d51, 75, 78, 0 },
  { 0x0bb6, 77, 79, 0 },
  { 0x0a40, 77, 48, 0 },
  { 0x5832, 80, 81, 1 }, /* 80 */
  { 0x4d1c, 88, 82, 0 },
  { 0x438e, 89, 83, 0 },
  { 0x3bdd, 90, 84, 0 },
  { 0x34ee, 91, 85, 0 },
  { 0x2eae, 92, 86, 0 }, /* 85 */
  { 0x299a, 93, 87, 0 },
  { 0x2516, 86, 71, 0 },
  { 0x5570, 88, 89, 1 },
  { 0x4ca9, 95, 90, 0 },
  { 0x44d9, 96, 91, 0 }, /* 90 */
  { 0x3e22, 97, 92, 0 },
  { 0x3824, 99, 93, 0 },
  { 0x32b4, 99, 94, 0 },
  { 0x2e17, 93, 86, 0 },
  { 0x56a8, 95, 96, 1 }, /* 95 */
  { 0x4f46,101, 97, 0 },
  { 0x47e5,102, 98, 0 },
  { 0x41cf,103, 99, 0 },
  { 0x3c3d,104,100, 0 },
  { 0x375e, 99, 93, 0 }, /* 100 */
  { 0x5231,105,102, 0 },
  { 0x4c0f,106,103, 0 },
  { 0x4639,107,104, 0 },
  { 0x415e,103, 99, 0 },
  { 0x5627,105,106, 1 }, /* 105 */
  { 0x50e7,108,107, 0 },
  { 0x4b85,109,103, 0 },
  { 0x5597,110,109, 0 },
  { 0x50ef,111,107, 0 },
  { 0x5a10,110,111, 1 }, /* 110 */
  { 0x5522,112,109, 0 },
  { 0x59eb,112,111, 1 }
};

#define a range0
#define c code
#define bp offset
#define ct cntr

static void
byte_out (Arith *coder)
{
  int t;

  t = coder->c >> 19;
  if (t > 0xff) {
    coder->data[coder->bp]++;
    while (coder->st) {
      coder->bp++;
      coder->data[coder->bp] = 0;
      coder->st--;
    }
    coder->bp++;
    coder->data[coder->bp] = t;
  } else {
    coder->bp++;
    coder->data[coder->bp] = t;
  }
}

static void
renorm_e (Arith *coder)
{
  do {
    coder->a <<= 1;
    coder->c <<= 1;
    coder->ct--;
    if (coder->ct == 0) {
      byte_out(coder);
      coder->ct = 8;
    }
  } while (coder->a < 0x8000);
}

static void
code_lps(Arith *coder, int s)
{
  int qe;
    
  qe = states[coder->contexts[s].state].qe_value;
  coder->a -= qe;
  if (coder->a >= qe) {
    coder->c += coder->a;
    coder->a = qe;
  }
  if(states[coder->contexts[s].state].switch_mps) {
    coder->contexts[s].mps ^= 1;
  }
  coder->contexts[s].state = states[coder->contexts[s].state].next_lps;
  renorm_e(coder);
}

static void
code_mps(Arith *coder, int s)
{
  int qe;

  qe = states[coder->contexts[s].state].qe_value;
  coder->a -= qe;

  if (coder->a < 0x8000) {
    if (coder->a < qe) {
      coder->c += coder->a;
      coder->a = qe;
    }
    coder->contexts[s].state = states[coder->contexts[s].state].next_mps;
    renorm_e(coder);
  }
}

static void
arith_qm_encode (Arith *coder, int s, int value)
{
  if (coder->contexts[s].mps == value) {
    code_mps(coder,s);
  } else {
    code_lps(coder,s);
  }
}

void
arith_qm_init (Arith *coder)
{
  coder->contexts[0].state = 0;
  coder->contexts[0].mps = 0;

  coder->st = 0;
  coder->a = 0x10000;
  coder->c = 0;
  coder->ct = 11;
  coder->bp = -1;
}

void
arith_qm_flush (Arith *coder)
{
  coder->bp++;

}

DEFINE_EFFICIENCY(qm)
DEFINE_SPEED(qm)

