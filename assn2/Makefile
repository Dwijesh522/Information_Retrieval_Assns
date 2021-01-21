prob:
	python3 prob_rerank.py ../data/msmarco-docdev-queries.tsv.gz ../data/msmarco-docdev-top100.gz ../data/docs.tsv.gz 1
lang:
	python3 lm_rerank.py ../data/msmarco-docdev-queries.tsv.gz ../data/msmarco-docdev-top100.gz ../data/docs.tsv.gz uni



#run:
#	python3 lm_rerank.py ../data/msmarco-docdev-queries.tsv.gz ../data/msmarco-docdev-top100.gz ../data/docs.tsv.gz uni
#	python3 prob_rerank.py ../data/msmarco-docdev-queries.tsv ../data/msmarco-docdev-top100 ../data/docs.tsv 1
#	python3 prob_rerank.py ../data/msmarco-docdev-queries.tsv ../data/msmarco-docdev-top100 ../data/docs.tsv 2
#	python3 prob_rerank.py ../data/msmarco-docdev-queries.tsv ../data/msmarco-docdev-top100 ../data/docs.tsv 3
#	python3 prob_rerank.py ../data/msmarco-docdev-queries.tsv ../data/msmarco-docdev-top100 ../data/docs.tsv 4
#	python3 prob_rerank.py ../data/msmarco-docdev-queries.tsv ../data/msmarco-docdev-top100 ../data/docs.tsv 5
#	python3 prob_rerank.py ../data/msmarco-docdev-queries.tsv ../data/msmarco-docdev-top100 ../data/docs.tsv 6
#	python3 prob_rerank.py ../data/msmarco-docdev-queries.tsv ../data/msmarco-docdev-top100 ../data/docs.tsv 7
#	python3 prob_rerank.py ../data/msmarco-docdev-queries.tsv ../data/msmarco-docdev-top100 ../data/docs.tsv 8
#	python3 prob_rerank.py ../data/msmarco-docdev-queries.tsv ../data/msmarco-docdev-top100 ../data/docs.tsv 9
#	python3 prob_rerank.py ../data/msmarco-docdev-queries.tsv ../data/msmarco-docdev-top100 ../data/docs.tsv 10

#qsub -P col764 -l select=2:ncpus=8:ngpus=1:mem=24G:centos=skylake -l walltime=6:00:00
