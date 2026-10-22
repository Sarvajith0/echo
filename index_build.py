import numpy as np, pandas as pd, torch, librosa, psycopg
from pgvector.psycopg import register_vector
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

MODEL_NAME = "laion/clap-htsat-unfused"
FMA_DIR = "data/fma_small"
N_TRACKS = 2000   # start small, raise to 5000+ once it works

# 1. Read the metadata (tracks.csv has a two-row header)
t = pd.read_csv("data/fma_metadata/tracks.csv", index_col=0, header=[0, 1])
t = t[t[("set", "subset")] == "small"]
t = t.sample(N_TRACKS, random_state=42)

# 2. Load model and database
model = ClapModel.from_pretrained(MODEL_NAME).eval()
proc = ClapProcessor.from_pretrained(MODEL_NAME)
conn = psycopg.connect("postgresql://echo:echo@localhost:5432/echo", autocommit=True)
register_vector(conn)
done = {r[0] for r in conn.execute("SELECT id FROM tracks")}  # lets you resume

def audio_path(tid):
    s = f"{tid:06d}"
    return f"{FMA_DIR}/{s[:3]}/{s}.mp3"

# 3. Embed each track
for tid, row in tqdm(t.iterrows(), total=len(t)):
    if tid in done:
        continue
    try:
        y, _ = librosa.load(audio_path(tid), sr=48000, mono=True)
        mid = len(y) // 2
        y = y[max(0, mid - 240000): mid + 240000]      # middle 10 seconds
        inputs = proc(audios=y, sampling_rate=48000, return_tensors="pt")
        with torch.no_grad():
            e = to_tensor(model.get_audio_features(**inputs)[0].numpy())
        e = e / np.linalg.norm(e)                      # normalise
        conn.execute(
            "INSERT INTO tracks VALUES (%s,%s,%s,%s,%s,%s)",
            (int(tid), row[("track", "title")], row[("artist", "name")],
             row[("track", "genre_top")], MODEL_NAME, e),
        )
    except Exception as ex:
        print("skipped", tid, ex)   # a few FMA files are corrupt, that's normal