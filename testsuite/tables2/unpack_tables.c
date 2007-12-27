
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <schroedinger/schro.h>
#include <schroedinger/schrounpack.h>


#define SHIFT 8
#define SIZE (1<<SHIFT)

int generate_table (void)
{
  uint8_t data[100];
  SchroUnpack unpack;
  int i,j,k;
  int x;
  int array[SHIFT*2];

  printf("\n");
  printf("#include <schroedinger/schrotables.h>\n");
  printf("\n");

  printf("const int schro_table_unpack_sint[%d][%d] = {\n", SIZE, 1+2*SHIFT);
  for(i=0;i<SIZE;i++){
    data[0] = (i<<(16-SHIFT))>>8;
    data[1] = (i<<(16-SHIFT))&0xff;
    data[2] = 0;
    data[3] = 0;
    memset (array, 0, SHIFT*2*sizeof(int));

    schro_unpack_init_with_data (&unpack, data, 4, 1);
    schro_unpack_limit_bits_remaining (&unpack, SHIFT);

    printf("  /* %3d */ { ", i);

    j = 0;
    x = schro_unpack_decode_sint (&unpack);
    while (schro_unpack_get_bits_read (&unpack) <= SHIFT) {
      array[j*2] = x;
      array[j*2+1] = schro_unpack_get_bits_read (&unpack);
      j++;
      x = schro_unpack_decode_sint (&unpack);
    }
    printf("%d, ", j);
    for(k=0;k<j;k++){
      printf("%d, %d, ", array[k*2], array[k*2+1]);
    }
    printf("},\n");
  }
  printf("};\n");

  return 1;
}


int
main (int argc, char *argv[])
{

  schro_init();

  generate_table();

  return 0;
}



