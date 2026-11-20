#!/usr/bin/env python3
"""
Project 8: Culinary Operations, Allergen & Recipe Manual Assistant
Knowledge Source: grand_bistro_culinary_recipe_manual_50p.pdf
"""

import sys
import json
from rag import answer, CulinaryRAGSystem

ASSESSMENT_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What exact quantities, in grams, of dark couverture chocolate and heavy cream are required for a standard 20-portion batch of Signature Truffle Ganache?",
        "expected_page": "Page 10"
    },
    {
        "id": "Q2",
        "question": "According to the Allergen Matrix, which dishes on the Autumn Tasting Menu contain tree nuts or sesame?",
        "expected_page": "Page 25"
    },
    {
        "id": "Q3",
        "question": "At what time does the Eiffel Tower in Paris close to visitors during the winter season?",
        "expected_page": None
    },
    {
        "id": "Q4",
        "question": "What is the mandatory minimum internal cooking temperature, in degrees Celsius, and the required holding time for sous-vide duck breast, as specified in the Food Safety section?",
        "expected_page": "Page 40"
    },
    {
        "id": "Q5",
        "question": "What is the current Michelin-star rating of Noma in Copenhagen?",
        "expected_page": None
    }
]

def run_assessment():
    print("=" * 80)
    print("PROJECT 8: CULINARY OPERATIONS, ALLERGEN & RECIPE MANUAL ASSISTANT")
    print("Corpus: grand_bistro_culinary_recipe_manual_50p.pdf (50 Pages)")
    print("=" * 80)
    print("\nInitializing RAG system and vector database...\n")
    
    # Initialize RAG system
    CulinaryRAGSystem.get_instance()
    
    print("\nRunning Assessment Questions (Q1 - Q5):\n")
    results = []
    
    for item in ASSESSMENT_QUESTIONS:
        qid = item["id"]
        q = item["question"]
        print("-" * 80)
        print(f"[{qid}] QUESTION: {q}")
        output = answer(q)
        print(f"[{qid}] ANSWER  : {output['answer']}")
        print(f"[{qid}] SOURCES : {output['sources']}")
        results.append({
            "id": qid,
            "question": q,
            "answer": output["answer"],
            "sources": output["sources"]
        })
        print()

    print("=" * 80)
    print("ASSESSMENT SUMMARY (JSON OUTPUT)")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    print("=" * 80)

def main():
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        res = answer(user_query)
        print(json.dumps(res, indent=2))
    else:
        run_assessment()

if __name__ == "__main__":
    main()
