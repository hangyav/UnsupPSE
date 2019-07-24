#!/bin/bash
source ./environment.sh

# ================= MKDIR ======================
mkdir -p $DATA

# ================= DOWNLOAD ===================
# NEWSCRAWL
# EN
echo -n "" > $DATA/news.2011-14.en
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.en.shuffled.gz | gunzip >> $DATA/news.2011-14.en
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz | gunzip >> $DATA/news.2011-14.en
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz | gunzip >> $DATA/news.2011-14.en
curl  http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.en.shuffled.v2.gz | gunzip >> $DATA/news.2011-14.en

# DE
echo -n "" > $DATA/news.2011-14.de
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.de.shuffled.gz | gunzip >> $DATA/news.2011-14.de
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.de.shuffled.gz | gunzip >> $DATA/news.2011-14.de
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.de.shuffled.gz | gunzip >> $DATA/news.2011-14.de
curl  http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.de.shuffled.v2.gz | gunzip >> $DATA/news.2011-14.de

# FR
echo -n "" > $DATA/news.2011-14.fr
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.fr.shuffled.gz | gunzip >> $DATA/news.2011-14.fr
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.fr.shuffled.gz | gunzip >> $DATA/news.2011-14.fr
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.fr.shuffled.gz | gunzip >> $DATA/news.2011-14.fr
curl  http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.fr.shuffled.v2.gz | gunzip >> $DATA/news.2011-14.fr

# RU
echo -n "" > $DATA/news.2011-14.ru
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.ru.shuffled.gz | gunzip >> $DATA/news.2011-14.ru
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.ru.shuffled.gz | gunzip >> $DATA/news.2011-14.ru
curl  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.ru.shuffled.gz | gunzip >> $DATA/news.2011-14.ru
curl  http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.ru.shuffled.v2.gz | gunzip >> $DATA/news.2011-14.ru

# BUCC TRAINING
for f in bucc2017-de-en.training-gold.tar.bz2 bucc2017-fr-en.training-gold.tar.bz2 bucc2017-ru-en.training-gold.tar.bz2; do
    curl https://comparable.limsi.fr/bucc2017/$f -o $DATA/$f
    tar xvjf $DATA/$f -C $DATA
done

# BUCC SAMPLE
for f in bucc2017-de-en.sample-gold.tar.bz2 bucc2017-fr-en.sample-gold.tar.bz2 bucc2017-ru-en.sample-gold.tar.bz2; do
    curl https://comparable.limsi.fr/bucc2017/$f -o $DATA/$f
    tar xvjf $DATA/$f -C $DATA
done;

# FILTERING DATA
curl http://www.statmt.org/wmt18/parallel-corpus-filtering-data/data.gz  | gunzip | head -n $FILTERING_NUM > $DATA/$FILTER_INPUT.tsv
cut -f 1 $DATA/$FILTER_INPUT.tsv > $DATA/$FILTER_INPUT.en
cut -f 2 $DATA/$FILTER_INPUT.tsv > $DATA/$FILTER_INPUT.de
cut -f 3 $DATA/$FILTER_INPUT.tsv > $DATA/$FILTER_INPUT.scores

# ================= PREPROCESSS ================
# TOKENIZE
for lang in en de fr ru; do
    $MOSES/tokenizer/tokenizer.perl -q -l $lang -a < $DATA/news.2011-14.$lang > $DATA/news.2011-14.$lang.tok &
done
wait

# TRAIN TRUECASE MODEL
for lang in en de fr ru; do
    $MOSES/recaser/train-truecaser.perl --model $DATA/news.2011-14.$lang.tok.tcm --corpus $DATA/news.2011-14.$lang.tok &
done
wait

# TRUECASE
for lang in en de fr ru; do
    $MOSES/recaser/truecase.perl --model $DATA/news.2011-14.$lang.tok.tcm < $DATA/news.2011-14.$lang.tok > $DATA/news.2011-14.$lang.tok.tc &
done
wait

# SAVING PREPROCESSED NEWSCRAWL 2011 for mining; batching by sentence length
# head -n 15437674 $DATA/news.2011-14.en.tok.tc | awk 'BEGIN{FS=" "} NF>=16 && NF <= 20' | awk 'BEGIN{OFS="\t"}{print NR,$0}' > $DATA/news.2011.16_20.en.tok.tc
head -n 15437674 $DATA/news.2011-14.en.tok.tc | awk 'BEGIN{FS=" "} NF>=21 && NF <= 25' | awk 'BEGIN{OFS="\t"}{print NR,$0}' > $DATA/news.2011.21_25.en.tok.tc
# head -n 15437674 $DATA/news.2011-14.en.tok.tc | awk 'BEGIN{FS=" "} NF>=26 && NF <= 30' | awk 'BEGIN{OFS="\t"}{print NR,$0}' > $DATA/news.2011.26_30.en.tok.tc
# head -n 15437674 $DATA/news.2011-14.en.tok.tc | awk 'BEGIN{FS=" "} NF>=31 && NF <= 35' | awk 'BEGIN{OFS="\t"}{print NR,$0}' > $DATA/news.2011.31_35.en.tok.tc
# head -n 15437674 $DATA/news.2011-14.en.tok.tc | awk 'BEGIN{FS=" "} NF>=36 && NF <= 40' | awk 'BEGIN{OFS="\t"}{print NR,$0}' > $DATA/news.2011.36_40.en.tok.tc
# head -n 16037788 $DATA/news.2011-14.de.tok.tc | awk 'BEGIN{FS=" "} NF>=16 && NF <= 20' | awk 'BEGIN{OFS="\t"}{print NR,$0}' > $DATA/news.2011.16_20.de.tok.tc
head -n 16037788 $DATA/news.2011-14.de.tok.tc | awk 'BEGIN{FS=" "} NF>=21 && NF <= 25' | awk 'BEGIN{OFS="\t"}{print NR,$0}' > $DATA/news.2011.21_25.de.tok.tc
# head -n 16037788 $DATA/news.2011-14.de.tok.tc | awk 'BEGIN{FS=" "} NF>=26 && NF <= 30' | awk 'BEGIN{OFS="\t"}{print NR,$0}' > $DATA/news.2011.26_30.de.tok.tc
# head -n 16037788 $DATA/news.2011-14.de.tok.tc | awk 'BEGIN{FS=" "} NF>=31 && NF <= 35' | awk 'BEGIN{OFS="\t"}{print NR,$0}' > $DATA/news.2011.31_35.de.tok.tc
# head -n 16037788 $DATA/news.2011-14.de.tok.tc | awk 'BEGIN{FS=" "} NF>=36 && NF <= 40' | awk 'BEGIN{OFS="\t"}{print NR,$0}' > $DATA/news.2011.36_40.de.tok.tc

# TRUECASE BUCC
for src_lang in de fr ru; do
    for trg_lang in en; do
        for lang in $src_lang $trg_lang; do
            for data in training sample; do
                paste <(cut -f 1 $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$lang) <(cut -f 2 $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$lang | $MOSES/tokenizer/tokenizer.perl -a -q -l $lang | $MOSES/recaser/truecase.perl --model $DATA/news.2011-14.$lang.tok.tcm) > $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$lang.tok.tc &
            done;
        done;
        wait;
    done;
done;

for lang in de en; do
    $MOSES/tokenizer/tokenizer.perl -a -q -l $lang < $DATA/$FILTER_INPUT.$lang | $MOSES/recaser/truecase.perl --model $DATA/news.2011-14.$lang.tok.tcm | awk 'BEGIN{OFS="\t"}{print NR,$0}' > $DATA/$FILTER_INPUT.$lang.tok.tc &
done;
wait;

# # CLEAN
# for lang in $TRG_LANGS $SRC_LANGS; do
#     rm $DATA/news.2011-14.$lang.tok.tcm
# done
