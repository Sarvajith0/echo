import psycopg
conn = psycopg.connect("postgresql://echo:echo@localhost:5432/echo", autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.execute("CREATE TABLE IF NOT EXISTS test (id serial, emb vector(3))")
conn.execute("INSERT INTO test (emb) VALUES ('[1,2,3]'), ('[9,9,9]')")
row = conn.execute("SELECT id FROM test ORDER BY emb <-> '[1,2,4]' LIMIT 1").fetchone()
print("Nearest id:", row[0])   # should print 1