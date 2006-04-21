
void
schro_arith_context_encode_uu (SchroArith *arith, int context, int value)
{
  int i;

  for(i=0;i<value;i++){
    schro_arith_context_binary_encode (arith, context, 0);
  }
  schro_arith_context_binary_encode (arith, context, 1);
}

void
schro_arith_context_encode_su (SchroArith *arith, int context, int value)
{
  int i;
  int sign;

  if (value==0) {
    schro_arith_context_binary_encode (arith, context, 1);
    return;
  }
  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  for(i=0;i<value;i++){
    schro_arith_context_binary_encode (arith, context, 0);
  }
  schro_arith_context_binary_encode (arith, context, 1);
  schro_arith_context_binary_encode (arith, context, sign);
}

void
schro_arith_context_encode_ut (SchroArith *arith, int context, int value, int max)
{
  int i;

  for(i=0;i<value;i++){
    schro_arith_context_binary_encode (arith, context, 0);
  }
  if (value < max) {
    schro_arith_context_binary_encode (arith, context, 1);
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
schro_arith_context_encode_uegol (SchroArith *arith, int context, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  for(i=0;i<n_bits - 1;i++){
    schro_arith_context_binary_encode (arith, context, 0);
  }
  schro_arith_context_binary_encode (arith, context, 1);
  for(i=0;i<n_bits - 1;i++){
    schro_arith_context_binary_encode (arith, context, (value>>(n_bits - 2 - i))&1);
  }
}

void
schro_arith_context_encode_segol (SchroArith *arith, int context, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  schro_arith_context_encode_uegol (arith, context, value);
  if (value) {
    schro_arith_context_binary_encode (arith, context, sign);
  }
}

void
schro_arith_context_encode_ue2gol (SchroArith *arith, int context, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  schro_arith_context_encode_uegol (arith, context, n_bits - 1);
  for(i=0;i<n_bits - 1;i++){
    schro_arith_context_binary_encode (arith, context, (value>>(n_bits - 2 - i))&1);
  }
}

void
schro_arith_context_encode_se2gol (SchroArith *arith, int context, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  schro_arith_context_encode_ue2gol (arith, context, value);
  if (value) {
    schro_arith_context_binary_encode (arith, context, sign);
  }
}

