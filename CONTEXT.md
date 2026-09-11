# Query — Customer Support Agent

An AI support agent that classifies customer messages, drafts grounded replies, and decides whether to auto-handle or escalate. In v0 it supports a single brand: SpotifyCares.

## Language

**Brand**:
The company the agent supports. In v0, exactly one: SpotifyCares.
_Avoid_: company, handle, account

**Interaction**:
A single customer issue: starts with a customer message and ends at its closure verdict — Resolved, Uncertain, or Unresolved. A returning customer's next message always starts a new Interaction.
_Avoid_: conversation, thread, ticket

**Customer Message**:
The inbound tweet from a customer that opens an Interaction. In v0 the agent sees only this first message.
_Avoid_: tweet, query

**Resolved Case**:
An Interaction with evidence of natural closure — an explicit customer acknowledgement, or a brand reply indicating the issue/action was completed.
_Avoid_: closed thread, finished conversation

**Uncertain Case**:
An Interaction whose visible ending shows no evidence the issue was actually solved — typically the customer stopped responding after the brand's reply, but also when the customer's final message neither confirms resolution nor continues the issue.
_Avoid_: resolved, unanswered

**Unresolved Case**:
An Interaction where the customer continues asking, or the brand indicated it could not help.
_Avoid_: open ticket

**Historical Case**:
A Resolved Case used as retrieval evidence when drafting a reply.
_Avoid_: example, record, sample

**Auto-Handle**:
The agent answers the customer itself, permitted only when evidence is sufficient.
_Avoid_: automation, self-service

**Escalate**:
Routing the message to a human with a stated reason.
_Avoid_: handoff, transfer

**High-Risk Category**:
A manually defined message class that always escalates regardless of classifier confidence: security/credentials, payment disputes, account deletion/data requests, abuse or legal threats.
_Avoid_: sensitive topic

**Golden Set**:
The hand-labelled evaluation examples built by us, with gold intent and gold auto/escalate labels.
_Avoid_: test set, ground truth (unqualified)

**RAG Pool**:
The ~3,000 sampled Interactions retrieval draws its evidence from; only their Resolved Cases become Historical Cases.
_Avoid_: sample, training set

**Holdout**:
The ~1,000 sampled Interactions reserved for the Golden Set and never retrieved as evidence.
_Avoid_: test set, held-out data
