#!/bin/bash
source ./environment.sh

# ================= MKDIR =====================
mkdir -p $EMBEDDINGS
for src_lang in $SRC_LANGS; do
	for trg_lang in $TRG_LANGS; do
		mkdir -p $BWE_EMBEDDINGS/$src_lang-$trg_lang
	done;
done;
mkdir -p $DICTIONARIES

# ================= FASTTEXT ===================
for lang in $TRG_LANGS $SRC_LANGS; do
    $FASTTEXT skipgram -input $DATA/$INPUT.$lang.tok.tc -output $EMBEDDINGS/$INPUT.$lang -dim $DIM -ws 5 -neg 5 -loss ns -t 0.0001 -minn 3 -maxn 6 -minCount 5 -thread $THREADS
done

# ================= MUSE_MAPPING ============================
for src_lang in $SRC_LANGS; do
    for trg_lang in $TRG_LANGS; do
        CUDA_VISIBLE_DEVICES=$GPUS $PYTHON $MUSE/unsupervised.py --src_lang $src_lang --tgt_lang $trg_lang --src_emb $EMBEDDINGS/$INPUT.$src_lang.vec --tgt_emb $EMBEDDINGS/$INPUT.$trg_lang.vec --n_epochs 5 --n_refinement 1 --emb_dim $DIM --exp_path $BWE_EMBEDDINGS --exp_name $src_lang-$trg_lang --exp_id muse_unsupervised --dico_eval ''
    done;
done;

# ================= BWE_DICTIONARY_GENERATION ===============
for src_lang in $SRC_LANGS; do
    for trg_lang in $TRG_LANGS; do
        CUDA_VISIBLE_DEVICES=$GPUS $PYTHON scripts/bilingual_nearest_neighbor.py --source_embeddings $BWE_EMBEDDINGS/$src_lang-$trg_lang/muse_unsupervised/vectors-$src_lang.txt --target_embeddings $BWE_EMBEDDINGS/$src_lang-$trg_lang/muse_unsupervised/vectors-$trg_lang.txt --output $DICTIONARIES/BWE.$src_lang-$trg_lang.sim --knn $TOPN_DICT -m csls --cslsknn $TOPN_CSLS --max_vocab 200000
    done;
done;

# ================= LEVENSTEIN_DICTIONARY_GENERATION ===============
for src_lang in $SRC_LANGS; do
    for trg_lang in $TRG_LANGS; do
        $PYTHON scripts/edit_distance_sym_delete.py -s <(tail -n+2 $EMBEDDINGS/$INPUT.$src_lang.vec | cut -f 1 -d ' ') -t <(tail -n+2 $EMBEDDINGS/$INPUT.$trg_lang.vec | cut -f 1 -d ' ') -o $DICTIONARIES/ORTH.$src_lang-$trg_lang.sim -th 0.8 -mif 1 -k 2
    done;
done;
