from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


class KnowledgeBase:

    def __init__(self):
        print("Loading knowledge base...")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.client = chromadb.PersistentClient(
            path="knowledge/chroma"
        )

        self.collection = self.client.get_or_create_collection(
            name="smart_chip"
        )

        self._load_knowledge()

    def _load_knowledge(self):
        knowledge_path = Path("knowledge/smart_chip.md")

        text = knowledge_path.read_text(
            encoding="utf-8"
        )

        chunks = self._chunk_text(text)

        # Don't duplicate existing documents
        existing = self.collection.get()

        if len(existing["ids"]) == len(chunks):
            print(f"Knowledge base ready ({len(chunks)} chunks).")
            return

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

    def _chunk_text(self, text, max_chars=800):
        paragraphs = text.split("\n\n")

        chunks = []
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
                current += "\n" + paragraph

        if current:
            chunks.append(current.strip())

        return chunks

    def search(self, query, n_results=2):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=embedding,
            n_results=n_results
        )

        return results["documents"][0]