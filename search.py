import numpy as np, torch, psycopg
from pgvector.psycopg import register_vector
from transformers import ClapModel, ClapProcessor

MODEL_NAME = "laion/clap-htsat-unfused"
model = ClapModel.from_pretrained(MODEL_NAME).eval()
proc = ClapProcessor.from_pretrained(MODEL_NAME)
conn = psycopg.connect("postgresql://echo:echo@localhost:5432/echo", autocommit=True)
register_vector(conn)

def search(query, k=10):
    inputs = proc(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        q = to_tensor(model.get_text_features(**inputs)[0].numpy())
    q = q / np.linalg.norm(q)
    return conn.execute(
        "SELECT id, title, artist, genre, emb <=> %s AS dist "
        "FROM tracks ORDER BY emb <=> %s LIMIT %s", (q, q, k)
    ).fetchall()

if __name__ == "__main__":
    for r in search("slow sad acoustic guitar"):
        print(r)