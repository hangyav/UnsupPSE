#!/bin/bash
source ./environment.sh
set -

# ================= PRINT_RESULTS ===========================
echo BUCC17
for mine_method in $MINE_METHODS; do
    echo -e "\t$mine_method"
    for filter_method in $FILTER_METHODS; do
        echo -e "\t\t$filter_method"
        for data in $BUCC_SETS; do
            echo -e "\t\t\t$data\t\tPRECISION\t\tRECALL\t\tF1"
            for src_lang in $SRC_LANGS; do
                for trg_lang in $TRG_LANGS; do
                    if [ -f "$MINING/bucc2017/$src_lang-$trg_lang/${mine_method}_${filter_method}_$src_lang-$trg_lang.$data.sim.pred.res" ]; then
                        echo -e -n "\t\t\t\t$src_lang-$trg_lang: "
                        tail -n 1 $MINING/bucc2017/$src_lang-$trg_lang/${mine_method}_${filter_method}_$src_lang-$trg_lang.$data.sim.pred.res
                    fi
                done;
            done;
        done;
    done;
done;
