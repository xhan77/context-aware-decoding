# nq series
SCENE="-1"
RTRUNC="2038"
GENLEN="10"

for MODEL in "llama-30B" #"gpt-neox-20b" # "gpt-neox-2.7b" "gpt-neox-20b" # "opt-iml-30b" "opt-30b"
do
    for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
    do
        for FILT in "1" # "0.8" "0.6" "0.4"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="0.0"
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/nq_swap/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
                done
            done
        done
    done
done

for MODEL in "llama-30B" #"gpt-neox-20b" # "gpt-neox-2.7b" "gpt-neox-20b" # "opt-iml-30b" "opt-30b"
do
    for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
    do
        for FILT in "1" # "0.8" "0.6" "0.4"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="0.0"
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/nq/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
                done
            done
        done
    done
done

for MODEL in "llama-30B" #"gpt-neox-20b" # "gpt-neox-2.7b" "gpt-neox-20b" # "opt-iml-30b" "opt-30b"
do
    for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
    do
        for FILT in "1" # "0.8" "0.6" "0.4"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="0.0"
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/memotrap/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
                done
            done
        done
    done
done