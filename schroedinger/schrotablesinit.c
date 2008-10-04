
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroutils.h>
#include <schroedinger/schrotables.h>

static int16_t *quantise_table;
static int16_t *dequantise_table_intra;
static int16_t *dequantise_table_inter;

void
schro_tables_init (void)
{
  int i,j;
  static int inited;

  if (inited) return;
  inited = TRUE;

  quantise_table = schro_malloc (61 * 65536 * sizeof(int16_t));
  dequantise_table_intra = schro_malloc (61 * 65536 * sizeof(int16_t));
  dequantise_table_inter = schro_malloc (61 * 65536 * sizeof(int16_t));

  for(i=0;i<61;i++){
    int quant_factor;
    int quant_offset_intra;
    int quant_offset_inter;

    quant_factor = schro_table_quant[i];
    quant_offset_intra = schro_table_offset_1_2[i];
    quant_offset_inter = schro_table_offset_3_8[i];

    for(j=0;j<65536;j++){
      quantise_table[i*65536 + j] =
        schro_quantise (j - 32768, quant_factor, 0);
      dequantise_table_intra[i*65536 + j] =
        schro_dequantise (j - 32768, quant_factor, quant_offset_intra);
      dequantise_table_inter[i*65536 + j] =
        schro_dequantise (j - 32768, quant_factor, quant_offset_inter);
    }
  }

}

int16_t *
schro_tables_get_quantise_table (int quant_index)
{
  return quantise_table + quant_index * 65536;
}

int16_t *
schro_tables_get_dequantise_table (int quant_index, schro_bool is_intra)
{
  if (is_intra) {
    return dequantise_table_intra + quant_index * 65536;
  } else {
    return dequantise_table_inter + quant_index * 65536;
  }
}
