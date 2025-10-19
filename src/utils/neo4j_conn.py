from neo4j import GraphDatabase
from .settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_write(cypher: str, params: dict | None=None):
    with driver.session(database="neo4j") as s:
        return s.run(cypher, params or {}).consume()

def run_query(cypher: str, params: dict | None=None):
    with driver.session(database="neo4j") as s:
        return list(s.run(cypher, params or {}))
