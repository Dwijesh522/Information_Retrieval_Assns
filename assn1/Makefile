invidx:
	@time python3 invidx_cons.py /home/dwijesh/Documents/sem7/ir/dataset/assignment1/TaggedTrainingAP indexfile
	
	@echo 'compressing the indexfile.dict...'
	@mv indexfile.dict deleteThis.dict
	@7z a indexfile.dict deleteThis.dict
	@rm deleteThis.dict
	
	@echo 'compressing the indexfile.idx...'
	@mv indexfile.idx deleteThis.idx
	@7z a indexfile.idx deleteThis.idx
	@rm deleteThis.idx

vecsearch:
	@echo 'extracting indexfile.dict file...'
	@7z e indexfile.dict
	@rm indexfile.dict
	@mv deleteThis.dict indexfile.dict
	@echo 'extracting indexfile.idx...'
	@7z e indexfile.idx
	@rm indexfile.idx
	@mv deleteThis.idx indexfile.idx
	@time python3 vecsearch.py --query /home/dwijesh/Documents/sem7/ir/dataset/assignment1/topics.51-100 --cutoff 150 --output resultfile --index /home/dwijesh/Documents/sem7/ir/assns/assn1/src/indexfile.idx --dict /home/dwijesh/Documents/sem7/ir/assns/assn1/src/indexfile.dict
	@echo 'compressing the indexfile.dict...'
	@mv indexfile.dict deleteThis.dict
	@7z a indexfile.dict deleteThis.dict
	@rm deleteThis.dict
	
	@echo 'compressing the indexfile.idx...'
	@mv indexfile.idx deleteThis.idx
	@7z a indexfile.idx deleteThis.idx
	@rm deleteThis.idx


score:
	@./trec_eval -m ndcg_cut.10 -m set_F -M100 ../../../dataset/assignment1/qrels.filtered.51-100 resultfile

printdict:
	@echo 'extracting indexfile.idx...'
	@7z e indexfile.idx
	@rm indexfile.idx
	@mv deleteThis.idx indexfile.idx
	@python3 printdict.py

compress:
	@echo 'compressing the indexfile.dict...'
	@mv indexfile.dict deleteThis.dict
	@7z a indexfile.dict deleteThis.dict
	@rm deleteThis.dict
	
	@echo 'compressing the indexfile.idx...'
	@mv indexfile.idx deleteThis.idx
	@7z a indexfile.idx deleteThis.idx
	@rm deleteThis.idx

extract:
	@echo 'extracting indexfile.dict file...'
	@7z e indexfile.dict
	@rm indexfile.dict
	@mv deleteThis.dict indexfile.dict
	@echo 'extracting indexfile.idx...'
	@7z e indexfile.idx
	@rm indexfile.idx
	@mv deleteThis.idx indexfile.idx
