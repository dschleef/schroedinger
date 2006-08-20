
#ifndef _SCHRO_BITSTREAM_H_
#define _SCHRO_BITSTREAM_H_

typedef enum _SchroParseCode SchroParseCode;
enum _SchroParseCode {
  SCHRO_PARSE_CODE_ACCESS_UNIT = 0x00,
  SCHRO_PARSE_CODE_INTRA_REF = 0x0c,
  SCHRO_PARSE_CODE_INTRA_NON_REF = 0x08,
  SCHRO_PARSE_CODE_INTER_REF_1 = 0x0d,
  SCHRO_PARSE_CODE_INTER_REF_2 = 0x0e,
  SCHRO_PARSE_CODE_INTER_NON_REF_1 = 0x09,
  SCHRO_PARSE_CODE_INTER_NON_REF_2 = 0x0a,
  SCHRO_PARSE_CODE_END_SEQUENCE = 0x10
};

#define SCHRO_PARSE_CODE_PICTURE(is_ref,n_refs) (8 | ((is_ref)<<2) | (n_refs))

#define SCHRO_PARSE_CODE_IS_PICTURE(x) ((x) & 0x8)
#define SCHRO_PARSE_CODE_NUM_REFS(x) ((x) & 0x3)
#define SCHRO_PARSE_CODE_IS_REF(x) ((x) & 0x4)

typedef struct _SchroSignalRange SchroSignalRange;
struct _SchroSignalRange {
  int luma_offset;
  int luma_excursion;
  int chroma_offset;
  int chroma_excursion;
};

enum _SchroColourMatrix {
  SCHRO_COLOUR_MATRIX_CUSTOM = 0,
  SCHRO_COLOUR_MATRIX_SDTV = 1,
  SCHRO_COLOUR_MATRIX_HDTV = 2,
  SCHRO_COLOUR_MATRIX_YCgCo = 3
};

enum _SchroColourPrimaries {
  SCHRO_COLOUR_PRIMARY_CUSTOM = 0,
  SCHRO_COLOUR_PRIMARY_NTSC = 1,
  SCHRO_COLOUR_PRIMARY_PAL = 2,
  SCHRO_COLOUR_PRIMARY_HDTV = 3
};

enum _SchroTransferChar {
  SCHRO_TRANSFER_CHAR_TV = 0,
  SCHRO_TRANSFER_CHAR_EXTENDED = 1,
  SCHRO_TRANSFER_CHAR_LINEAR = 2
};

#endif

