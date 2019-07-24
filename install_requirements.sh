conda install faiss-gpu==1.5.3 cudatoolkit==9.0 pytorch=1.1.0 torchvision -c pytorch -y
conda install numpy==1.16.4 tqdm==4.19.5 nltk==3.2.4 -y
pip install  gensim==3.2.0 python-levenshtein==0.12.0 langid==1.1.6

python -c 'import nltk; nltk.download("stopwords")'
