# ================= BASH =======================================================
# print commands
set -x
# exit in case of error
set -e
# ================= EXECUTABLES ================================================
PYTHON=python
# PYTHON=~/.anaconda/envs/pse/bin/python
FASTTEXT=./third_party/fastText/fasttext
MUSE=./third_party/MUSE
MOSES=./third_party/moses/scripts

# ================= DATA =======================================================
DATA=./data
# IN CASE OF OTHER LANGUAGES than en de fr ru make sure to download monolingual data and preprocess them like in get_data.sh
INPUT=news.2011-14 # data for creating dictionaries; this is extended with src and trg language ids

# PARAMETERS FOR BUCC EXPERIMENTS
SRC_LANGS='de fr ru' # which BUCC language pairs to run; each source language is paired with each target language
TRG_LANGS='en'
BUCC_SETS='sample training' # list of bucc datasets

# PARAMETERS FOR GENERAL SENTENCE EXTRACTION
EXTRACT_INPUT=news.2011.21_25 # dataset for extracting sentences; see get_data.sh for length based batching and preparation; set custom mining parameters below
EXTRACT_SRC_LANG=de
EXTRACT_TRG_LANG=en

# PARAMETERS FOR FILTERING
FILTER_INPUT=filtering_data # dataset for extracting sentences; see get_data.sh for length based batching and preparation; set custom mining parameters below
FILTERING_NUM=5000000 # limit the number of lines from the original data since it is large (16G); use -0 to download all
FILTER_SRC_LANG=de
FILTER_TRG_LANG=en

RESULTS=./results
EMBEDDINGS=$RESULTS/embeddings
BWE_EMBEDDINGS=$EMBEDDINGS/bwe
DOC_EMBEDDINGS=$EMBEDDINGS/doc
DICTIONARIES=$RESULTS/dictionaries
MINING=$RESULTS/mining
FILTERING=$RESULTS/filtering

# ============= MINING PARAMETERS =============================================
THREADS=20 # number of threads to use
GPUS=0 # comma separated list of visible GPU indices
DIM=300 # dimension of embeddings
TOPN_DICT=100 # topn most similar translation candidates in the generated BWE dictionary
TOPN_CSLS=500 # topn most similar neighbors used for CSLS
TOPN_DOC=100 # topn most similar document candidates for prefiltering

# BUCC
MINE_METHODS='maxalign max' # list of mining methods; supported: max (averaging most similar words); maxalign (subsegment detection)
FILTER_METHODS='static dynamic' # list of filtering methods; supported: static; dynamic

# EXTRACTION
EXTRACT_MINE_METHOD='maxalign' # mining method; supported one of: max (averging most similar words); maxalign (subsegment detection)
EXTRACT_FILTER_METHOD='static' # filtering method; supported: static; dynamic

# FILTERING
FILTER_MINE_METHOD='maxalign' # scoring method; supported one of: max (averging most similar words); maxalign (subsegment detection)
FILTER_TOP=1000000 # keep top pairs from original data


declare -A MINE_PARAMS # parameters for mining; these were tuned on bucc sample in case of bucc
# maximum length difference of aligned segments; minimum length of segments; similarity threshold for segment detection; window size for similarity smoothing
MINE_PARAMS[bucc17_maxalign_de_en]='--max_length_diff 5  --min_length 0.5 --threshold 0.25 --window_size 15 --remove_stopwords 1'
MINE_PARAMS[bucc17_maxalign_fr_en]='--max_length_diff 5  --min_length 0.5 --threshold 0.25 --window_size 20 --remove_stopwords 1'
MINE_PARAMS[bucc17_maxalign_ru_en]='--max_length_diff 5  --min_length 0.2 --threshold 0.25 --window_size 15 --remove_stopwords 1'
MINE_PARAMS[bucc17_max_de_en]='--remove_stopwords 1'
MINE_PARAMS[bucc17_max_fr_en]='--remove_stopwords 1'
MINE_PARAMS[bucc17_max_ru_en]='--remove_stopwords 1'
# default values for extraction
MINE_PARAMS[${EXTRACT_INPUT}_maxalign_${EXTRACT_SRC_LANG}_${EXTRACT_TRG_LANG}]='--max_length_diff 5  --min_length 0.7 --threshold 0.3 --window_size 5 --remove_stopwords 1'
MINE_PARAMS[${EXTRACT_INPUT}_max_${EXTRACT_SRC_LANG}_${EXTRACT_TRG_LANG}]='--remove_stopwords 1'

declare -A FILTER_THRESHOLDS # threshold values for extracting sentences from scored pairs
# what similarity value to consider as parallel (tuned on BUCC sample dataset)
FILTER_THRESHOLDS[bucc17_maxalign_static_de_en]='0.31'
FILTER_THRESHOLDS[bucc17_maxalign_static_fr_en]='0.37'
FILTER_THRESHOLDS[bucc17_maxalign_static_ru_en]='0.14'
FILTER_THRESHOLDS[bucc17_max_static_de_en]='0.36'
FILTER_THRESHOLDS[bucc17_max_static_fr_en]='0.42'
FILTER_THRESHOLDS[bucc17_max_static_ru_en]='0.23'
FILTER_THRESHOLDS[bucc17_maxalign_dynamic_de_en]='1.5'
FILTER_THRESHOLDS[bucc17_maxalign_dynamic_fr_en]='1.5'
FILTER_THRESHOLDS[bucc17_maxalign_dynamic_ru_en]='1.5'
FILTER_THRESHOLDS[bucc17_max_dynamic_de_en]='2.0'
FILTER_THRESHOLDS[bucc17_max_dynamic_fr_en]='2.0'
FILTER_THRESHOLDS[bucc17_max_dynamic_ru_en]='2.0'
# default values for extraction
FILTER_THRESHOLDS[${EXTRACT_INPUT}_${EXTRACT_MINE_METHOD}_static_${EXTRACT_SRC_LANG}_${EXTRACT_TRG_LANG}]='0.3'
FILTER_THRESHOLDS[${EXTRACT_INPUT}_${EXTRACT_MINE_METHOD}_dynamic_${EXTRACT_SRC_LANG}_${EXTRACT_TRG_LANG}]='1.5'


# default values for corpus filtering
declare -A CORPUS_FILTER_PARAMS # parameters for corpus filtering
CORPUS_FILTER_PARAMS[${FILTER_INPUT}_pre_${FILTER_SRC_LANG}_${FILTER_TRG_LANG}]='--min_sent_length 10 --max_sent_len_diff 15 --number_url_ratio_threshold 0.6'
CORPUS_FILTER_PARAMS[${FILTER_INPUT}_maxalign_${FILTER_SRC_LANG}_${FILTER_TRG_LANG}]='--similarity_weight 1.0 --edit_similarity_weight 0.2 --max_length_diff 5  --min_length 0.5 --threshold 0.25 --window_size 15 --remove_stopwords 1'
CORPUS_FILTER_PARAMS[${FILTER_INPUT}_max_${FILTER_SRC_LANG}_${FILTER_TRG_LANG}]='--similarity_weight 1.0 --edit_similarity_weight 0.2 --remove_stopwords 1'
