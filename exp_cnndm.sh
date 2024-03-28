# cnndm
GLOBALLEN="2048"
MAXCTXLEN="1948"
GENLEN="100"
FN_PREFIX="eval/cnndm_example_input/cnndm"

for WEIGHT in "1_0" "1.5_-0.5"
do
    TOPP="0.9"
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    bash run_group_decode_fileio.sh 2023 "0,1" $TESTFILE $GLOBALLEN $MAXCTXLEN $GENLEN $TOPP
done
