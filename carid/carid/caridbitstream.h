
#ifndef _CARID_BITSTREAM_H_
#define _CARID_BITSTREAM_H_

typedef enum _CaridParseCode CaridParseCode;
enum _CaridParseCode {
  CARID_PARSE_CODE_RAP = 0xd7,
  CARID_PARSE_CODE_INTRA_REF = 0xd1,
  CARID_PARSE_CODE_INTRA_NON_REF = 0xd2,
  CARID_PARSE_CODE_INTER_REF = 0xd3,
  CARID_PARSE_CODE_INTER_NON_REF = 0xd4,
  CARID_PARSE_CODE_SEQUENCE_STOP = 0xd0,
  CARID_PARSE_CODE_DATA = 0xff
};

typedef struct _CaridSignalRange CaridSignalRange;
struct _CaridSignalRange {
  int bit_depth;
  int accuracy_bits;
  int luma_offset;
  int luma_excursion;
  int chroma_offset;
  int chroma_excursion;
};

enum _CaridColourMatrix {
  CARID_COLOUR_MATRIX_CUSTOM = 0,
  CARID_COLOUR_MATRIX_SDTV = 1,
  CARID_COLOUR_MATRIX_HDTV = 2,
  CARID_COLOUR_MATRIX_YCgCo = 3
};

enum _CaridColourPrimaries {
  CARID_COLOUR_PRIMARY_CUSTOM = 0,
  CARID_COLOUR_PRIMARY_NTSC = 1,
  CARID_COLOUR_PRIMARY_PAL = 2,
  CARID_COLOUR_PRIMARY_HDTV = 3
};

enum _CaridTransferChar {
  CARID_TRANSFER_CHAR_TV = 0,
  CARID_TRANSFER_CHAR_EXTENDED = 1,
  CARID_TRANSFER_CHAR_LINEAR = 2
};

#endif

