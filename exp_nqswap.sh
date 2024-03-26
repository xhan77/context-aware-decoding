# nq-swap
GLOBALLEN="2048"
MAXCTXLEN="2038"
GENLEN="10"
FN_PREFIX="nqswap_example_input/nqswap"

for WEIGHT in "1_0" "2_-1"
do
    TOPP="0.0"
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    bash run_group_decode_fileio.sh 2023 "0,1" $TESTFILE $GLOBALLEN $MAXCTXLEN $GENLEN $TOPP
done
