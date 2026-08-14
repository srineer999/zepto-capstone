import os
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi import FastAPI
from langgraph.graph import StateGraph, END


# ============================================================
# 1. LOCAL EMBEDDING MODEL + CHROMADB
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, "chroma_db")
)

collection = client.get_or_create_collection(
    name="zepto_policies"
)


def load_documents():
    documents = []
    ids = []

    print("Documents folder:", DOCS_DIR)

    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.endswith(".txt"):
            path = os.path.join(DOCS_DIR, filename)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if text:
                documents.append(text)
                ids.append(filename)

    print("Documents found:", ids)

    if documents:
        embeddings = embedder.encode(documents).tolist()

        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )

    print(f"Loaded {len(documents)} policy documents.")
    print("ChromaDB document count:", collection.count())


load_documents()


# ============================================================
# 2. MOCK LLM
# ============================================================

MOCK_LLM = True


def mock_llm(prompt):
    return "Based on the retrieved policy context: " + prompt


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    context: str
    sources: list
    confidence: float
    answer: str


# ============================================================
# 4. NODE 1 — CLASSIFY INTENT
# ============================================================

def classify_intent(state: AssistantState):

    query = state["query"].lower()

    policy_words = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "track",
        "cancel",
        "cancellation",
        "gift card",
        "giftcard",
        "support",
        "hours",
        "damaged",
        "missing",
        "replacement",
        "priority",
        "pass"
    ]

    if any(word in query for word in policy_words):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {"intent": intent}


# ============================================================
# 5. NODE 2 — RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(state: AssistantState):

    query = state["query"]

    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]

    context = "\n\n".join(docs)

    prompt = (
        "Answer the customer question using only the retrieved "
        "Zepto policy context.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}"
    )

    answer = mock_llm(prompt)

    return {
        "context": context,
        "sources": ids,
        "confidence": 1.0 if docs else 0.0,
        "answer": answer
    }


# ============================================================
# 6. NODE 3 — DIRECT ANSWER
# ============================================================

def direct_answer(state: AssistantState):

    return {
        "answer": (
            "I can help with Zepto delivery, returns, refunds, "
            "membership, tracking, cancellation, gift cards, "
            "damaged or missing items, and customer support policies."
        ),
        "sources": [],
        "confidence": 1.0
    }


# ============================================================
# 7. CONDITIONAL ROUTING
# ============================================================

def route_question(state: AssistantState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# 8. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AssistantState)

graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve_and_answer", retrieve_and_answer)
graph.add_node("direct_answer", direct_answer)

graph.set_entry_point("classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph.add_edge("retrieve_and_answer", END)
graph.add_edge("direct_answer", END)

assistant_graph = graph.compile()


# ============================================================
# 9. PYDANTIC OUTPUT
# ============================================================

class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list
    confidence: float


# ============================================================
# 10. FASTAPI
# ============================================================

app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    result = assistant_graph.invoke(
        {"query": request.query}
    )

    return AskResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0)
    )


# ============================================================
# 11. LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_questions = [
        "What is the refund policy?",
        "How can I track my order?"
    ]

    for question in test_questions:

        result = assistant_graph.invoke(
            {"query": question}
        )

        print("\nQUESTION:", question)
        print("ANSWER:", result.get("answer"))
        print("SOURCES:", result.get("sources"))
        print("CONFIDENCE:", result.get("confidence"))

    print("\nSUPPORT ASSISTANT READY!")