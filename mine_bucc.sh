#!/bin/bash
source ./environment.sh

# ================= MKDIR =====================
for src_lang in $SRC_LANGS; do
	for trg_lang in $TRG_LANGS; do
		mkdir -p $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang
		mkdir -p $MINING/bucc2017/$src_lang-$trg_lang
	done;
done;
mkdir -p $DICTIONARIES

# ================= AVERAGED DOCUMENT REPRESENTATION ===============
for data in $BUCC_SETS; do
    for src_lang in $SRC_LANGS; do
        for trg_lang in $TRG_LANGS; do
            for lang in $src_lang $trg_lang; do
                $PYTHON scripts/doc2vec.py --input $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$lang.tok.tc --output $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$lang.vec --embeddings $BWE_EMBEDDINGS/$src_lang-$trg_lang/muse_unsupervised/vectors-$lang.txt --stopword_language $lang
            done;
        done;
    done;
done;

# ================= DOC_DICTIONARY_GENERATION_FOR_PREFILTERING ===============
for data in $BUCC_SETS; do
    for src_lang in $SRC_LANGS; do
        for trg_lang in $TRG_LANGS; do
            CUDA_VISIBLE_DEVICES=$GPUS $PYTHON scripts/bilingual_nearest_neighbor.py --source_embeddings $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$src_lang.vec --target_embeddings $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$trg_lang.vec --output $DICTIONARIES/DOC.$src_lang-$trg_lang.$data.sim --knn $TOPN_DOC -m nn
        done;
    done;
done;

# ================= DOCUMENT_SIMILARITIES ===================
for mine_method in $MINE_METHODS; do
    for data in $BUCC_SETS; do
        for src_lang in $SRC_LANGS; do
            for trg_lang in $TRG_LANGS; do
                $PYTHON scripts/doc_similarity.py -s $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$src_lang.tok.tc -t $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$trg_lang.tok.tc -o $MINING/bucc2017/$src_lang-$trg_lang/${mine_method}_$src_lang-$trg_lang.$data.sim -sim $DICTIONARIES/BWE.$src_lang-$trg_lang.sim -dsim $DICTIONARIES/DOC.$src_lang-$trg_lang.$data.sim -n $THREADS -fl $src_lang -tl $trg_lang -esim $DICTIONARIES/ORTH.$src_lang-$trg_lang.sim -m ${mine_method} ${MINE_PARAMS[bucc17_${mine_method}_${src_lang}_${trg_lang}]}
            done;
        done;
    done;
done;

# ================= MINING_&_EVALUATION ==============================
for mine_method in $MINE_METHODS; do
    for filter_method in $FILTER_METHODS; do
        for data in $BUCC_SETS; do
            for src_lang in $SRC_LANGS; do
                for trg_lang in $TRG_LANGS; do
                    $PYTHON ./scripts/filter.py -i $MINING/bucc2017/$src_lang-$trg_lang/${mine_method}_$src_lang-$trg_lang.$data.sim -m $filter_method -th ${FILTER_THRESHOLDS[bucc17_${mine_method}_${filter_method}_${src_lang}_${trg_lang}]} -o $MINING/bucc2017/$src_lang-$trg_lang/${mine_method}_${filter_method}_$src_lang-$trg_lang.$data.sim.pred
                    $PYTHON scripts/bucc_f-score.py  -p $MINING/bucc2017/$src_lang-$trg_lang/${mine_method}_${filter_method}_$src_lang-$trg_lang.$data.sim.pred -g $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.gold > $MINING/bucc2017/$src_lang-$trg_lang/${mine_method}_${filter_method}_$src_lang-$trg_lang.$data.sim.pred.res
                done;
            done;
        done;
    done;
done;
