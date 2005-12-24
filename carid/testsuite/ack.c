
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

