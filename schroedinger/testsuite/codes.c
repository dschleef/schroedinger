
#include <stdio.h>

#include <carid/carid.h>



void
carid_arith_context_binary_encode (CaridArith *arith, int i, int value)
{
  printf(" %d", value);
}

#if 0

void
carid_arith_context_encode_uu (CaridArith *arith, int context, int value)
{
  int i;

  for(i=0;i<value;i++){
    carid_arith_context_binary_encode (arith, context, 0);
  }
  carid_arith_context_binary_encode (arith, context, 1);
}

void
carid_arith_context_encode_su (CaridArith *arith, int context, int value)
{
  int i;
  int sign;

  if (value==0) {
    carid_arith_context_binary_encode (arith, context, 1);
    return;
  }
  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  for(i=0;i<value;i++){
    carid_arith_context_binary_encode (arith, context, 0);
  }
  carid_arith_context_binary_encode (arith, context, 1);
  carid_arith_context_binary_encode (arith, context, sign);
}

void
carid_arith_context_encode_ut (CaridArith *arith, int context, int value, int max)
{
  int i;

  for(i=0;i<value;i++){
    carid_arith_context_binary_encode (arith, context, 0);
  }
  if (value < max) {
    carid_arith_context_binary_encode (arith, context, 1);
  }
}

static int
maxbit (unsigned int x)
{
  int i;
  for(i=0;x;i++){
    x >>= 1;
  }
  return i;
}

void
carid_arith_context_encode_uegol (CaridArith *arith, int context, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  for(i=0;i<n_bits - 1;i++){
    carid_arith_context_binary_encode (arith, context, 0);
  }
  carid_arith_context_binary_encode (arith, context, 1);
  for(i=0;i<n_bits - 1;i++){
    carid_arith_context_binary_encode (arith, context, (value>>(n_bits - 2 - i))&1);
  }
}

void
carid_arith_context_encode_segol (CaridArith *arith, int context, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  carid_arith_context_encode_uegol (arith, context, value);
  if (value) {
    carid_arith_context_binary_encode (arith, context, sign);
  }
}

void
carid_arith_context_encode_ue2gol (CaridArith *arith, int context, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  carid_arith_context_encode_uegol (arith, context, n_bits - 1);
  for(i=0;i<n_bits - 1;i++){
    carid_arith_context_binary_encode (arith, context, (value>>(n_bits - 2 - i))&1);
  }
}

void
carid_arith_context_encode_se2gol (CaridArith *arith, int context, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  carid_arith_context_encode_ue2gol (arith, context, value);
  if (value) {
    carid_arith_context_binary_encode (arith, context, sign);
  }
}

#endif


int
main (int argc, char *argv[])
{
  int i;

  carid_init();

  printf("unsigned unary\n");
  for(i=0;i<5;i++) {
    printf("%3d:", i);
    carid_arith_context_encode_uu(NULL,0,0,i);
    printf("\n");
  }
  printf("\n");

  printf("signed unary\n");
  for(i=-5;i<6;i++) {
    printf("%3d:", i);
    carid_arith_context_encode_su(NULL,0,i);
    printf("\n");
  }
  printf("\n");

  printf("unsigned truncated unary (n=4)\n");
  for(i=0;i<5;i++) {
    printf("%3d:", i);
    carid_arith_context_encode_ut(NULL,0,i,4);
    printf("\n");
  }
  printf("\n");

  printf("unsigned exp-Golomb\n");
  for(i=0;i<11;i++) {
    printf("%3d:", i);
    carid_arith_context_encode_uegol(NULL,0,i);
    printf("\n");
  }
  printf("\n");

  printf("signed exp-Golomb\n");
  for(i=-5;i<6;i++) {
    printf("%3d:", i);
    carid_arith_context_encode_segol(NULL,0,i);
    printf("\n");
  }
  printf("\n");

  printf("unsigned exp-exp-Golomb\n");
  for(i=0;i<11;i++) {
    printf("%3d:", i);
    carid_arith_context_encode_ue2gol(NULL,0,i);
    printf("\n");
  }
  printf("\n");

  printf("signed exp-exp-Golomb\n");
  for(i=-5;i<6;i++) {
    printf("%3d:", i);
    carid_arith_context_encode_se2gol(NULL,0,i);
    printf("\n");
  }
  printf("\n");

  return 0;

}



