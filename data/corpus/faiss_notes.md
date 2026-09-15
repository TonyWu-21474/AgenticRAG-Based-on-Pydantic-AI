# FAISS notes

FAISS is a library for efficient similarity search over dense vectors. This
project uses `faiss.IndexFlatIP`, a brute-force inner-product index, because it
is exact and the corpus is small.

## Cosine similarity from inner product

Inner product equals cosine similarity when every vector is L2-normalised.
The indexing path therefore calls `faiss.normalize_L2` on the embedding matrix
before adding it to the index, and calls it again on the query vector before
searching. Scores returned by `index.search` then land in the range -1 to 1.

## Persistence

`faiss.write_index` and `faiss.read_index` serialise the index to disk as
`index.faiss`. Chunk metadata is written separately to `chunks.json` with the
embedding model name and vector dimension, which lets the loader detect an
index built with a different embedding model.

## Scaling up

`IndexFlatIP` scans every vector, so it is O(n) per query. Past a few hundred
thousand chunks the usual upgrade is `IndexHNSWFlat` for approximate search or
`IndexIVFFlat` with a training step. Both keep the same add/search API, so only
the index construction changes.

## Practical defaults

- 1024-dimensional vectors from the `bge-m3` embedding model.
- Chunks of roughly 900 characters with 150 characters of overlap.
- `top_k` between 4 and 8 is usually enough context for one answer.
