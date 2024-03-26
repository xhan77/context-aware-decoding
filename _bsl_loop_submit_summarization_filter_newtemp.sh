# xsum
SCENE="-1"
RTRUNC="1948"
GENLEN="100"

for MODEL in "gpt-neo-2.7b" "opt-13b" "flan-t5-xl" "flan-t5-xxl" # "opt-2.7b" "opt-350m" "opt-6.7b" "opt-1.3b" "opt-125m" "flan-t5-base" "flan-t5-large" "opt-iml-1.3b"
do
    # "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
    for WEIGHT in "3.5_-0.17" "3.67_-0.33" "3.83_-0.5" "4.0_-0.67" "4.17_-0.83" #"10.0_-6.67" "16.67_-13.33" "3.33_0.0" "33.33_-30.0" "4.33_-1.0" "5.0_-1.67" "5.67_-2.33" "6.67_-3.33" "66.67_-63.33" "8.33_-5.0"
    do
        for FILT in "1"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="1.0" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/xsum/all_new${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel.sbatch
                done
            done
        done
    done
done

for MODEL in "opt-30b" # "opt-iml-30b"
do
    for WEIGHT in "3.5_-0.17" "3.67_-0.33" "3.83_-0.5" "4.0_-0.67" "4.17_-0.83" #"10.0_-6.67" "16.67_-13.33" "3.33_0.0" "33.33_-30.0" "4.33_-1.0" "5.0_-1.67" "5.67_-2.33" "6.67_-3.33" "66.67_-63.33" "8.33_-5.0"
    do
        for FILT in "1"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="1.0" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/xsum/all_new${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
                done
            done
        done
    done
done

for MODEL in "gpt-neox-20b"
do
    for WEIGHT in "3.5_-0.17" "3.67_-0.33" "3.83_-0.5" "4.0_-0.67" "4.17_-0.83" #"10.0_-6.67" "16.67_-13.33" "3.33_0.0" "33.33_-30.0" "4.33_-1.0" "5.0_-1.67" "5.67_-2.33" "6.67_-3.33" "66.67_-63.33" "8.33_-5.0"
    do
        for FILT in "1"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="1.0" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/xsum/all_new${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
                done
            done
        done
    done
done

for MODEL in "llama-13B" "llama-7B"
do
    for WEIGHT in "3.5_-0.17" "3.67_-0.33" "3.83_-0.5" "4.0_-0.67" "4.17_-0.83" #"10.0_-6.67" "16.67_-13.33" "3.33_0.0" "33.33_-30.0" "4.33_-1.0" "5.0_-1.67" "5.67_-2.33" "6.67_-3.33" "66.67_-63.33" "8.33_-5.0"
    do
        for FILT in "1"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="1.0" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/xsum/all_new${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel.sbatch
                done
            done
        done
    done
done

for MODEL in "llama-30B"
do
    for WEIGHT in "3.5_-0.17" "3.67_-0.33" "3.83_-0.5" "4.0_-0.67" "4.17_-0.83" #"10.0_-6.67" "16.67_-13.33" "3.33_0.0" "33.33_-30.0" "4.33_-1.0" "5.0_-1.67" "5.67_-2.33" "6.67_-3.33" "66.67_-63.33" "8.33_-5.0"
    do
        for FILT in "1"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="1.0" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/xsum/all_new${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
                done
            done
        done
    done
done

# cnndm
for MODEL in "gpt-neo-2.7b" "opt-13b" "flan-t5-xl" "flan-t5-xxl" # "opt-2.7b" "opt-350m" "opt-6.7b" "opt-1.3b" "opt-125m" "flan-t5-base" "flan-t5-large" "opt-iml-1.3b"
do
    # "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
    for WEIGHT in "3.5_-0.17" "3.67_-0.33" "3.83_-0.5" "4.0_-0.67" "4.17_-0.83" #"10.0_-6.67" "16.67_-13.33" "3.33_0.0" "33.33_-30.0" "4.33_-1.0" "5.0_-1.67" "5.67_-2.33" "6.67_-3.33" "66.67_-63.33" "8.33_-5.0"
    do
        for FILT in "1"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="1.0" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/cnndm/all_new${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnlab.sbatch
                done
            done
        done
    done
done

for MODEL in "opt-30b" # "opt-iml-30b"
do
    for WEIGHT in "3.5_-0.17" "3.67_-0.33" "3.83_-0.5" "4.0_-0.67" "4.17_-0.83" #"10.0_-6.67" "16.67_-13.33" "3.33_0.0" "33.33_-30.0" "4.33_-1.0" "5.0_-1.67" "5.67_-2.33" "6.67_-3.33" "66.67_-63.33" "8.33_-5.0"
    do
        for FILT in "1"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="1.0" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/cnndm/all_new${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
                done
            done
        done
    done
done

for MODEL in "gpt-neox-20b"
do
    for WEIGHT in "3.5_-0.17" "3.67_-0.33" "3.83_-0.5" "4.0_-0.67" "4.17_-0.83" #"10.0_-6.67" "16.67_-13.33" "3.33_0.0" "33.33_-30.0" "4.33_-1.0" "5.0_-1.67" "5.67_-2.33" "6.67_-3.33" "66.67_-63.33" "8.33_-5.0"
    do
        for FILT in "1"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="1.0" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/cnndm/all_new${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
                done
            done
        done
    done
done

for MODEL in "llama-13B" "llama-7B"
do
    for WEIGHT in "3.5_-0.17" "3.67_-0.33" "3.83_-0.5" "4.0_-0.67" "4.17_-0.83" #"10.0_-6.67" "16.67_-13.33" "3.33_0.0" "33.33_-30.0" "4.33_-1.0" "5.0_-1.67" "5.67_-2.33" "6.67_-3.33" "66.67_-63.33" "8.33_-5.0"
    do
        for FILT in "1"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="1.0" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/cnndm/all_new${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnlab.sbatch
                done
            done
        done
    done
done

for MODEL in "llama-30B"
do
    for WEIGHT in "3.5_-0.17" "3.67_-0.33" "3.83_-0.5" "4.0_-0.67" "4.17_-0.83" #"10.0_-6.67" "16.67_-13.33" "3.33_0.0" "33.33_-30.0" "4.33_-1.0" "5.0_-1.67" "5.67_-2.33" "6.67_-3.33" "66.67_-63.33" "8.33_-5.0"
    do
        for FILT in "1"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="1.0" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/cnndm/all_new${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
                done
            done
        done
    done
done