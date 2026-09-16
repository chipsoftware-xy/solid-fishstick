from pathlib import Path
import re
import chromadb
from sentence_transformers import SentenceTransformer


class KnowledgeBase:

    def __init__(self, knowledge_file="employee_id.md"):
        print("Loading knowledge base...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="knowledge/chroma")
        self.collection = self.client.get_or_create_collection(
        name=f"smart_chip_{knowledge_file.replace('.md', '')}")
        self.knowledge_file = knowledge_file
        self._load_knowledge()

    def _load_knowledge(self):
        knowledge_path = Path("knowledge") / self.knowledge_file

        text = knowledge_path.read_text(
            encoding="utf-8"
        )

        chunks = self._chunk_text(text)

        # Compare actual CONTENT, not just chunk count. The old check
        # (`len(existing) == len(chunks)`) meant editing the file without
        # changing the number of paragraph-chunks would silently never
        # trigger a reindex — Chroma would keep serving stale embeddings
        # forever. Sorting handles the case where insertion order isn't
        # guaranteed to match.
        existing = self.collection.get()
        existing_docs = existing.get("documents") or []

        if existing_docs and sorted(existing_docs) == sorted(chunks):
            print(f"Knowledge base ready ({len(chunks)} chunks).")
            return

        if existing["ids"]:
            print("Knowledge file changed — rebuilding index...")
            self.collection.delete(ids=existing["ids"])

        print(f"Indexing {len(chunks)} knowledge chunks...")

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True
        ).tolist()

        ids = [
            f"smart_chip_{i}"
            for i in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

        print(
            f"Knowledge base ready ({len(chunks)} chunks)."
        )

    def _chunk_text(self, text, max_chars=1500):
        """
        Each "## " section becomes its own chunk, unconditionally — no
        grouping of multiple sections into one chunk at all.

        Confirmed as a real cause of a content-retrieval miss: an
        earlier version grouped sections together up to a char limit,
        which still merged a focused section like "Access control &
        biometrics" with unrelated neighbors (printers, consumables,
        etc.), diluting its embedding enough that a narrow query like
        "biometric identification" didn't score within max_distance —
        even though the file explicitly does cover biometrics. Keeping
        every section fully separate removes that dilution entirely
        rather than just reducing it. This file is small enough that
        the resulting handful of extra chunks costs nothing meaningful
        at retrieval time.

        max_chars now only guards against a single section being
        unexpectedly huge (split at a paragraph boundary in that case,
        not mid-sentence) — it no longer controls normal chunking at
        all when sections are reasonably sized, which they are here.
        """
        raw_sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
        sections = [s.strip() for s in raw_sections if s.strip()]

        chunks = []
        for section in sections:
            if len(section) <= max_chars:
                chunks.append(section)
                continue
            # Fallback for an unexpectedly long section — split on
            # paragraph breaks within it rather than dropping this
            # safeguard entirely.
            paragraphs = section.split("\n\n")
            current = ""
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                if len(current) + len(paragraph) > max_chars:
                    if current:
                        chunks.append(current.strip())
                    current = paragraph
                else:
                    current = f"{current}\n{paragraph}" if current else paragraph
            if current:
                chunks.append(current.strip())

        return chunks

    def search(self, query, n_results=2, max_distance=1.0, debug=False):
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=embedding,
            n_results=n_results,
            include=["documents", "distances"],
        )

        docs = results["documents"][0]
        distances = results["distances"][0]

        if debug:
            for doc, dist in zip(docs, distances):
                print(f"[KB] distance={dist:.3f} :: {doc[:60]!r}")

        # Chroma always returns n_results regardless of relevance — a
        # vague query like "yes" would previously still get back
        # whatever chunk was closest, even if it wasn't actually
        # relevant, and that got fed to the LLM as fact. Filtering by
        # distance means a poor match returns nothing instead of noise.
        # max_distance is untuned — print with debug=True on a few real
        # queries (both relevant and vague) and adjust based on what you
        # see; 1.0 is a reasonable starting point for normalized
        # embeddings with Chroma's default L2 metric.
        return [
            doc for doc, dist in zip(docs, distances)
            if dist <= max_distance
        ]