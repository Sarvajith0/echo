import psycopg
conn = psycopg.connect("postgresql://echo:echo@localhost:5432/echo", autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.execute("""
CREATE TABLE IF NOT EXISTS tracks (
    id        integer PRIMARY KEY,
    title     text,
    artist    text,
    genre     text,
    model     text,          -- which embedding model made the vector
    emb       vector(512)
)""")