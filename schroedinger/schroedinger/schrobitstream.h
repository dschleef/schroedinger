
#ifndef _SCHRO_BITSTREAM_H_
#define _SCHRO_BITSTREAM_H_

typedef enum _SchroParseCode SchroParseCode;
enum _SchroParseCode {
  SCHRO_PARSE_CODE_RAP = 0xd7,
  SCHRO_PARSE_CODE_INTRA_REF = 0xd1,
  SCHRO_PARSE_CODE_INTRA_NON_REF = 0xd2,
  SCHRO_PARSE_CODE_INTER_REF = 0xd3,
  SCHRO_PARSE_CODE_INTER_NON_REF = 0xd4,
  SCHRO_PARSE_CODE_SEQUENCE_STOP = 0xd0,
  SCHRO_PARSE_CODE_DATA = 0xff
};

typedef struct _SchroSignalRange SchroSignalRange;
struct _SchroSignalRange {
  int bit_depth;
  int accuracy_bits;
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

