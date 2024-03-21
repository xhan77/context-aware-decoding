# nq-swap
MAXCTXLEN="2038"
GENLEN="10"
FN_PREFIX="nqswap_example_input/nqswap"

for WEIGHT in "1_0" # "2_-1"
do
    TOPP="0.0"
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    bash bsl_testrun_decode_ar_contrastive_fileio.sh 2023 "0,1" 404 404 "n/a" $TESTFILE 404 404 404 "bsl_test_ensemble" 0 $MAXCTXLEN $GENLEN 0.0 $TOPP "n/a"
done
