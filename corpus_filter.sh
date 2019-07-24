#!/bin/bash
source ./environment.sh

# ================= MKDIR =====================
mkdir -p $DOC_EMBEDDINGS
mkdir -p $FILTERING
mkdir -p $DICTIONARIES

# ================= PRE-FILTERING ==================================
$PYTHON scripts/corpus_pre_filter.py -s $DATA/$FILTER_INPUT.$FILTER_SRC_LANG.tok.tc -t $DATA/$FILTER_INPUT.$FILTER_TRG_LANG.tok.tc -sc $DATA/$FILTER_INPUT.scores -o $FILTERING/$FILTER_INPUT.pre.scores ${CORPUS_FILTER_PARAMS[${FILTER_INPUT}_pre_${FILTER_SRC_LANG}_${FILTER_TRG_LANG}]} -n $THREADS -fl $FILTER_SRC_LANG -tl $FILTER_TRG_LANG

# ===================== FILTERING ==================================
$PYTHON scripts/corpus_filter.py  -s $DATA/$FILTER_INPUT.$FILTER_SRC_LANG.tok.tc -t $DATA/$FILTER_INPUT.$FILTER_TRG_LANG.tok.tc -sc $FILTERING/$FILTER_INPUT.pre.scores -o $FILTERING/$FILTER_INPUT.scores -sim $DICTIONARIES/BWE.$FILTER_SRC_LANG-$FILTER_TRG_LANG.sim -esim $DICTIONARIES/ORTH.$FILTER_SRC_LANG-$FILTER_TRG_LANG.sim -m ${FILTER_MINE_METHOD} ${CORPUS_FILTER_PARAMS[${FILTER_INPUT}_${FILTER_MINE_METHOD}_${FILTER_SRC_LANG}_${FILTER_TRG_LANG}]} -n $THREADS

# ===================== EXTRACT ====================================
paste $FILTERING/$FILTER_INPUT.scores $DATA/$FILTER_INPUT.$FILTER_SRC_LANG.tok.tc $DATA/$FILTER_INPUT.$FILTER_TRG_LANG.tok.tc | egrep -v '^-?0.0000' | sort -n -r | head -n $FILTER_TOP > $FILTERING/$FILTER_INPUT.out
