import json
import os
from proto13 import retrieve, generate, save_feedback

# Clear old feedback log for a clean test
if os.path.exists("feedback_log.json"):
    os.remove("feedback_log.json")

# Define a fixed set of test queries
test_queries = [
    "What is artificial intelligence?",
    "Explain machine learning basics.",
    "How does reinforcement learning work?",
]

print("=== Before Feedback ===")
before_results = {}
for q in test_queries:
    context, docs = retrieve(q)
    answer = generate(q, context)
    before_results[q] = {"docs": docs, "answer": answer}
    print(f"Query: {q}")
    print(f"Docs used: {docs}")
    print(f"Answer: {answer}\n")

# Simulate feedback - thumbs up on first doc, thumbs down on second doc (if exists)
for q, result in before_results.items():
    docs_used = result["docs"]
    if docs_used:
        thumbs = "up"
        # Simulate positive feedback for first doc only, negative if multiple docs
        save_feedback({
            "query": q,
            "answer": result["answer"],
            "thumbs": thumbs,
            "docs_used": [docs_used[0]],
            "edited": ""
        })
        if len(docs_used) > 1:
            save_feedback({
                "query": q,
                "answer": result["answer"],
                "thumbs": "down",
                "docs_used": [docs_used[1]],
                "edited": ""
            })

print("\n=== After Feedback ===")
for q in test_queries:
    context, docs = retrieve(q)
    answer = generate(q, context)
    print(f"Query: {q}")
    print(f"Docs used: {docs}")
    print(f"Answer: {answer}\n")

