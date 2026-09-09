# Hiver SDE Intern — AI Support Agent

## Technical Specification — v0

### 1. Objective

Build an AI support agent for **one brand** from the Customer Support on Twitter dataset.

The agent must:

1. Classify incoming customer messages into brand-specific intents.
2. Draft a reply grounded in historically resolved support conversations.
3. Decide whether to **auto-handle** or **escalate to a human**, with a reason.
4. Provide measurable evidence that the system works.

---

## 2. High-Level Architecture

```text
Customer Message
       │
       ▼
┌─────────────────────┐
│ Intent Classifier    │
│ Small model / LLM    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Historical Retrieval │
│ BM25 + Embeddings    │
└──────────┬──────────┘
           │
       Top K cases
           │
           ▼
┌─────────────────────┐
│ Reply Generator      │
│ LLM                   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Decision Policy       │
│ Auto / Escalate       │
└──────────┬──────────┘
       ┌───┴────┐
       ▼        ▼
     AUTO    ESCALATE
```

---

# 3. Data Pipeline

### Input

Customer Support on Twitter dataset.

### Processing

* Select one brand.
* Reconstruct customer ↔ brand conversations.
* Remove unusable/incomplete records.
* Identify resolved support interactions.
* Sample a manageable development/evaluation dataset.

### Output

A normalized case format:

```json
{
  "conversation_id": "...",
  "customer_message": "...",
  "brand_reply": "...",
  "intent": "...",
  "resolution": "..."
}
```

---

# 4. Intent Classifier

### Goal

Map each customer message to a small taxonomy derived from the selected brand's data.

Target: approximately **8–15 intents**.

### Approach

Compare:

* **Baseline:** TF-IDF + Logistic Regression
* **System:** small LLM or lightweight embedding-based classifier

### Output

```json
{
  "intent": "refund_status",
  "confidence": 0.91
}
```

Evaluate using:

* Accuracy
* Macro-F1
* Per-intent F1
* Confusion matrix

---

# 5. Historical Retrieval / RAG

### Goal

Find previous support cases that are relevant to the current customer issue.

### Approach

Build an index over historical resolved cases.

Use:

* BM25 for lexical similarity
* Embeddings for semantic similarity
* Optional weighted/hybrid ranking

Retrieve top **3–5 cases**.

Prefer cases matching the predicted intent.

### Output

```text
Customer message
+
Predicted intent
+
Top K historical cases
```

These become evidence for reply generation.

---

# 6. Reply Generator

### Goal

Generate a useful response consistent with the brand's historical support behavior.

### Model

LLM with a structured prompt.

### Inputs

* Customer message
* Predicted intent
* Retrieved historical cases
* Relevant historical replies/resolutions

### Requirements

The generated reply should be:

* Relevant
* Grounded
* Actionable
* Brand-consistent
* Non-hallucinatory

The model must **not invent policies, refunds, timelines, or capabilities** unsupported by the evidence.

### Output

```json
{
  "reply": "...",
  "evidence": ["case_123", "case_456"]
}
```

---

# 7. Auto-Handle vs Escalate

Start with a **deterministic policy**, rather than another LLM.

Escalate when:

* Intent confidence is below threshold.
* Retrieval evidence is weak.
* No sufficiently similar historical resolution exists.
* Issue requires account-specific investigation.
* Issue falls into a manually defined high-risk category.

Otherwise:

```text
AUTO-HANDLE
```

Output:

```json
{
  "decision": "escalate",
  "reason": "Insufficient historical evidence to safely resolve this issue."
}
```

Later, optionally compare this policy against an LLM-based decision-maker.

---

# 8. Evaluation

Create a **150–250 example human-labelled golden set**.

Each example should contain:

```text
customer_message
gold_intent
gold_auto/escalate
optional reference cases
human notes
```

Evaluate three dimensions:

### Intent

Accuracy + Macro-F1

### Reply

LLM-as-judge using a rubric covering:

* Correctness
* Relevance
* Grounding
* Actionability
* Brand consistency

Validate the judge against human ratings on a subset.

### Escalation

Measure:

* Accuracy
* Precision/Recall
* False-auto rate

False auto-handling is especially important.

---

# 9. Baselines

At minimum:

### Baseline 1 — Trivial

* Majority-class intent
* Generic support reply
* Always auto or always escalate

### Baseline 2 — Simple

* TF-IDF + Logistic Regression
* Similarity-based historical reply retrieval
* Rule-based escalation

Compare both against the proposed system.

---

# 10. Failure Analysis

Identify the **top 5 failure modes**.

For each:

```text
Failure
Example
Why it happened
Hypothesis
Potential fix
```

Examples may include:

* Ambiguous intents
* Poor retrieval
* Hallucinated information
* Incorrect escalation
* Rare intents
* Conversation context loss

---

# 11. Headline Metrics

Report a small set of headline numbers:

```text
Intent Macro-F1
Reply quality score
False-auto rate
Auto-handling rate
```

Also include:

> **What is misleading about my headline number?**

Explicitly discuss dataset imbalance, evaluation limitations, leakage, judge reliability, or other factors that could make the headline result look better than real-world performance.

---

# 12. Repository Structure

```text
hiver-support-agent/
│
├── data/
├── src/
│   ├── preprocessing/
│   ├── classification/
│   ├── retrieval/
│   ├── generation/
│   ├── escalation/
│   └── evaluation/
│
├── tests/
├── golden_set/
├── notebooks/
├── configs/
├── README.md
├── REPORT.md
└── requirements.txt
```

The README must allow the headline evaluation to be reproduced in **<15 minutes** on the provided subset.

---

# 13. Key Design Principle

The system should be **evidence-first**:

```text
Understand the issue
        ↓
Find what the brand historically did
        ↓
Generate a grounded response
        ↓
Only automate when evidence is sufficient
```

The primary objective is not maximum automation.

It is **safe, measurable, evidence-backed automation**.
