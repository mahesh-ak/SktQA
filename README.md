# NLU in Classical Languages

- Requires .env file with below format to access the LLMs and Neo4j KG:

```
OPENAI_API_KEY= "..."

FIREWORKS_API_KEY="..."

NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="neo4j"
NEO4J_DATABASE="neo4j"
```
- pip requirements are provided in requirements.txt
- Additionally Neo4j is needed 

- To generate run all the experiments and generate the results:

```make all```

- Knowledge graphs are by default loaded into database named ```neo4j```

Results and tables are generated in ```results/``` except ```results.json``` in the main folder

### Datasets and Models

- The Sanskrit Question-Answering dataset is available at [mahesh27/SktQA](https://huggingface.co/datasets/mahesh27/SktQA)
- The Sanskrit lemmatizer model is available at [mahesh27/t5lemmatizer](https://huggingface.co/mahesh27/t5lemmatizer)


### Citation

```
@inproceedings{akavarapu-etal-2025-case,
    title = "A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in {LLM}s",
    author = "Akavarapu, V.S.D.S.Mahesh  and
      Terdalkar, Hrishikesh  and
      Bhattacharyya, Pramit  and
      Agarwal, Shubhangi  and
      Deulgaonkar, Dr. Vishakha  and
      Dangarikar, Chaitali  and
      Manna, Pralay  and
      Bhattacharya, Arnab",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.141/",
    doi = "10.18653/v1/2025.findings-acl.141",
    pages = "2745--2761",
    ISBN = "979-8-89176-256-5",
}
```
