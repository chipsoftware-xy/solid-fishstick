from app.knowledge_base import KnowledgeBase


kb = KnowledgeBase()

while True:

    query = input("\nSearch: ")

    if query.lower() == "exit":
        break

    results = kb.search(query)

    print("\n--- RESULTS ---")

    for i, result in enumerate(results, 1):
        print(f"\n[{i}]")
        print(result)