# cnndm
MAXCTXLEN="1948"
GENLEN="100"
FN_PREFIX="cnndm_example_input/cnndm"

for WEIGHT in "1_0" "1.5_-0.5"
do
    TOPP="0.9"
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    bash bsl_testrun_decode_ar_contrastive_fileio.sh 2023 "0,1" 404 404 "n/a" $TESTFILE 404 404 404 "bsl_test_ensemble" 0 $MAXCTXLEN $GENLEN 0.0 $TOPP "n/a"
done
