
#include <stdio.h>
#include <liboil/liboil.h>
#include <liboil/liboilrandom.h>

#include "arith_qm.h"

unsigned char data[10000];

int
main (int argc, char *argv[])
{
  int i;
  int x;
  ArithQM coder = { 0 };
  int value;

  oil_init();

  for(x = 0; x <= 128; x += 1) {
    arith_qm_init (&coder);
    coder.data = data;
    for(i=0;i<10000;i++) {
      value = (oil_rand_u8() < x);
      arith_qm_encode (&coder, 0, value);
    }
    arith_qm_flush(&coder);

#if 0
    for(i=0;i<coder.bp;i++){
      printf("%02x\n", data[i]);
    }
#endif
    printf("%d %d\n", x, 8*coder.bp);
  }

  return 0;
}




