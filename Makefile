all: ramayana ayurveda ner mt evaluate

ramayana:
	python SktQA/zero_shot_qa.py
	python SktQA/rag_qa.py -k 4 -e bm25 -d ramayana
	python SktQA/rag_qa.py -k 4 -e bm25 -d sanskrit_en
	python SktQA/rag_qa.py -m gpt-4o
	sudo neo4j stop
	sudo neo4j-admin database import full neo4j --nodes=data/kg/ramayana/nodes_processed.csv --relationships=data/kg/ramayana/relationships_processed.csv --overwrite-destination
	sudo neo4j start
	python SktQA/kg_qa.py -i data/qa_set/sanskrit.tsv

ayurveda:
	python SktQA/rag_qa.py -d ayurveda -k 4 -e bm25
	python SktQA/rag_qa.py -d ayurveda_en -k 4 -e bm25
	python SktQA/rag_qa.py -d ayurveda -m gpt-4o
	sudo neo4j stop
	sudo neo4j-admin database import full neo4j --nodes=data/kg/ayurveda/nodes_processed.csv --relationships=data/kg/ayurveda/relationships_processed.csv --overwrite-destination
	sudo neo4j start
	python SktQA/kg_qa.py -i data/qa_set/ayurveda.tsv

ner:
	python SktQA/ner.py

mt:
	python SktQA/mt.py

evaluate:
	python SktQA/eval.py -k
	python SktQA/nlu_eval.py

