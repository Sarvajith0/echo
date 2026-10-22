from search import search

QUERIES = [
    ("heavy distorted guitars and loud drums", "Rock"),
    ("rapping over a hard beat with bass", "Hip-Hop"),
    ("acoustic guitar and soft folk singing", "Folk"),
    ("fast electronic dance music with synthesizers", "Electronic"),
    ("catchy upbeat pop song with vocals", "Pop"),
    ("calm instrumental music with no vocals", "Instrumental"),
    ("strange noisy experimental sounds", "Experimental"),
]

scores = []
for q, genre in QUERIES:
    hits = [r[3] for r in search(q, k=10)]
    p = sum(g == genre for g in hits) / 10
    scores.append(p)
    print(f"{p:.1f}  {q}")
print("Average genre precision@10:", sum(scores) / len(scores))