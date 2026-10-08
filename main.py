"""Tiny embedding similarity search utility."""
import csv
import argparse
import math

def read_embeddings(path):
    with open(path) as f:
        return [(row[0], [float(x) for x in row[1:]]) for row in csv.reader(f)]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def find_k(embeddings, query, k):
    return sorted(embeddings, key=lambda e: cosine(e[1], query), reverse=True)[:k]

def main():
    parser = argparse.ArgumentParser(description="Search nearest embeddings.")
    parser.add_argument("file", help="CSV file of embeddings (id, dim1, dim2, ...)")
    parser.add_argument("query", help="Comma-separated query vector")
    parser.add_argument("-k", type=int, default=5, help="Number of nearest neighbors")
    args = parser.parse_args()

    embeddings = read_embeddings(args.file)
    query = [float(x) for x in args.query.split(",")]
    results = find_k(embeddings, query, args.k)
    for id_, _ in results:
        print(id_)

if __name__ == "__main__":
    main()