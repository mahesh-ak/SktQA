all: ramayana ayurveda

ramayana:
	python SktQA/zero_shot_qa.py -r
	python SktQA/rag_qa.py -k 4 -e bm25
	python SktQA/rag_qa.py -k 4
	python SktQA/rag_qa.py -m gpt-4o

ayurveda:
	python SktQA/zero_shot_qa.py -i data/qa_set/ayurveda.tsv -r
	python SktQA/rag_qa.py -d ayurveda -k 4 -e bm25
	python SktQA/rag_qa.py -d ayurveda -k 4
	python SktQA/rag_qa.py -d ayurveda -m gpt-4o

