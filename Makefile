all: ramayana ayurveda evaluate

ramayana:
	python SktQA/zero_shot_qa.py -r
	python SktQA/rag_qa.py -k 4 -e bm25
	python SktQA/rag_qa.py -k 4
	python SktQA/rag_qa.py -m gpt-4o
	sudo neo4j stop
	sudo neo4j-admin database import full neo4j --nodes=data/kg/ramayana/nodes_processed.csv --relationships=data/kg/ramayana/relationships_processed.csv --overwrite-destination
	sudo neo4j start
	python SktQA/kg_qa.py -i data/qa_set/sanskrit.tsv

ayurveda:
	python SktQA/zero_shot_qa.py -i data/qa_set/ayurveda.tsv -r
	python SktQA/rag_qa.py -d ayurveda -k 4 -e bm25
	python SktQA/rag_qa.py -d ayurveda -k 4
	python SktQA/rag_qa.py -d ayurveda -m gpt-4o
	sudo neo4j stop
	sudo neo4j-admin database import full neo4j --nodes=data/kg/ayurveda/nodes_processed.csv --relationships=data/kg/ayurveda/relationships_processed.csv --overwrite-destination
	sudo neo4j start
	python SktQA/kg_qa.py -i data/qa_set/ayurveda.tsv


evaluate:
	python SktQA/evaluate.py -z
	python SktQA/evaluate.py -r
	python SktQA/evaluate.py -k

