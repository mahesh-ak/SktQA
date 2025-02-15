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


