# xsum
SCENE="-1"
RTRUNC="1948"
GENLEN="100"

# for MODEL in "gpt-neo-2.7b" #"opt-1.3b" "opt-125m" "opt-13b" "opt-2.7b" "opt-350m" "opt-6.7b" #"flan-t5-base" "flan-t5-large" "flan-t5-xl" "flan-t5-xxl" "opt-iml-1.3b"
# do
#     # "0_1" "1.5_-0.5" "1_0" "2_-1" "3_-2" "4_-3" "-1_2" "0.5_0.5"
#     for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
#     do
#         for FILT in "1" # "0.8" "0.6" "0.4"
#         do
#             for FN in "test.jsonl"
#             do
#                 for MODE in ""
#                 do
#                     TOPP="0.9" # 0.0, 0.9
#                     TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/xsum/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
#                     sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel.sbatch
#                 done
#             done
#         done
#     done
# done

# for MODEL in "opt-30b" #"opt-iml-30b"
# do
#     # "0_1" "1.5_-0.5" "1_0" "2_-1" "3_-2" "4_-3" "-1_2" "0.5_0.5"
#     for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
#     do
#         for FILT in "1" # "0.8" "0.6" "0.4"
#         do
#             for FN in "test.jsonl"
#             do
#                 for MODE in ""
#                 do
#                     TOPP="0.9" # 0.0, 0.9
#                     TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/xsum/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
#                     sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
#                 done
#             done
#         done
#     done
# done

# for MODEL in "gpt-neox-20b"
# do
#     # "0_1" "1.5_-0.5" "1_0" "2_-1" "3_-2" "4_-3" "-1_2" "0.5_0.5"
#     for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
#     do
#         for FILT in "1" # "0.8" "0.6" "0.4"
#         do
#             for FN in "test.jsonl"
#             do
#                 for MODE in ""
#                 do
#                     TOPP="0.9" # 0.0, 0.9
#                     TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/xsum/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
#                     sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
#                 done
#             done
#         done
#     done
# done

for MODEL in "llama-13B" #"llama-7B"
do
    for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
    do
        for FILT in "1" # "0.8" "0.6" "0.4"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="0.9" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/xsum/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel.sbatch
                done
            done
        done
    done
done

# for MODEL in "llama-30B"
# do
#     for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
#     do
#         for FILT in "1" # "0.8" "0.6" "0.4"
#         do
#             for FN in "test.jsonl"
#             do
#                 for MODE in ""
#                 do
#                     TOPP="0.9" # 0.0, 0.9
#                     TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/xsum/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
#                     sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
#                 done
#             done
#         done
#     done
# done

# cnndm
# for MODEL in "gpt-neo-2.7b" #"opt-1.3b" "opt-125m" "opt-13b" "opt-2.7b" "opt-350m" "opt-6.7b" #"flan-t5-base" "flan-t5-large" "flan-t5-xl" "flan-t5-xxl" "opt-iml-1.3b"
# do
#     # "0_1" "1.5_-0.5" "1_0" "2_-1" "3_-2" "4_-3" "-1_2" "0.5_0.5"
#     for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
#     do
#         for FILT in "1" # "0.8" "0.6" "0.4"
#         do
#             for FN in "test.jsonl"
#             do
#                 for MODE in ""
#                 do
#                     TOPP="0.9" # 0.0, 0.9
#                     TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/cnndm/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
#                     sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel.sbatch
#                 done
#             done
#         done
#     done
# done

# for MODEL in "opt-30b" #"opt-iml-30b"
# do
#     # "0_1" "1.5_-0.5" "1_0" "2_-1" "3_-2" "4_-3" "-1_2" "0.5_0.5"
#     for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
#     do
#         for FILT in "1" # "0.8" "0.6" "0.4"
#         do
#             for FN in "test.jsonl"
#             do
#                 for MODE in ""
#                 do
#                     TOPP="0.9" # 0.0, 0.9
#                     TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/cnndm/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
#                     sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
#                 done
#             done
#         done
#     done
# done

# for MODEL in "gpt-neox-20b"
# do
#     # "0_1" "1.5_-0.5" "1_0" "2_-1" "3_-2" "4_-3" "-1_2" "0.5_0.5"
#     for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
#     do
#         for FILT in "1" # "0.8" "0.6" "0.4"
#         do
#             for FN in "test.jsonl"
#             do
#                 for MODE in ""
#                 do
#                     TOPP="0.9" # 0.0, 0.9
#                     TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/cnndm/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
#                     sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
#                 done
#             done
#         done
#     done
# done

for MODEL in "llama-13B" #"llama-7B"
do
    for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
    do
        for FILT in "1" # "0.8" "0.6" "0.4"
        do
            for FN in "test.jsonl"
            do
                for MODE in ""
                do
                    TOPP="0.9" # 0.0, 0.9
                    TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/cnndm/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
                    sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel.sbatch
                done
            done
        done
    done
done

# for MODEL in "llama-30B"
# do
#     for WEIGHT in "2_-1" "3_-2" "2.5_-1.5" "0.5_0.5" "5_-4" "10_-9" "20_-19" "1.5_-0.5" "1_0"
#     do
#         for FILT in "1" # "0.8" "0.6" "0.4"
#         do
#             for FN in "test.jsonl"
#             do
#                 for MODE in ""
#                 do
#                     TOPP="0.9" # 0.0, 0.9
#                     TESTFILE="fin|/private/home/swj0419/contrast_decoding/data/cnndm/all${MODE}/${MODEL}/${WEIGHT}/filter_${FILT}/${FN}"
#                     sbatch --export=ALL,TESTFILE=$TESTFILE,RTRUNC=$RTRUNC,GENLEN=$GENLEN,TOPP=$TOPP bsl_submit_template_learnaccel_biginfer.sbatch
#                 done
#             done
#         done
#     done
# done