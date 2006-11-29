
#include <check.h>
#include <schroedinger/schro.h>
#include <stdlib.h>


START_TEST(test_decoder_init)
{
  SchroDecoder *decoder;

  decoder = schro_decoder_new();
  fail_unless (decoder != NULL);

  schro_decoder_free (decoder);
}
END_TEST

int
main (int argc, char *argv[])
{
  int number_failed;
  Suite *s;
  SRunner *sr;
  TCase *tc;

  schro_init();

  s = suite_create ("schro");

  tc = tcase_create ("schro");
  tcase_add_test (tc, test_decoder_init);
  suite_add_tcase (s, tc);

  sr = srunner_create (s);
  srunner_run_all (sr, CK_NORMAL);
  number_failed = srunner_ntests_failed (sr);
  srunner_free (sr);

  return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

