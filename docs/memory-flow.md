# How the agent uses memory

Two loops. The agent never changes; only what is in its prompt changes.

## Loop A — answering a question (read from memory)

```mermaid
flowchart TD
    Q["Question<br/><i>'Who are the top 5 customers?'</i>"] --> EMB[Embed the question<br/>1024 numbers]

    EMB --> SEARCH[(pgvector<br/>cosine search)]
    MEM[("feedback_memory<br/>stored lessons")] -.-> SEARCH

    SEARCH --> FILTER{"Filter"}
    FILTER -->|"similarity &lt; 0.45"| DROP1[discard<br/>not related enough]
    FILTER -->|"wrong embedding model"| DROP2[discard<br/>different vector space]
    FILTER -->|"status REJECTED"| DROP3[discard<br/>known bad]
    FILTER -->|survives| RANK

    RANK["Rank<br/>similarity 55%<br/>+ status 25%<br/>+ confidence 10%<br/>+ table overlap 10%"] --> TOPK[Top 5 lessons]

    TOPK --> SPLIT{Each lesson is used twice}
    SPLIT --> TABLES["1. Its tables are added<br/>to the schema the agent sees<br/><i>without this, a lesson about<br/>refunds is unusable</i>"]
    SPLIT --> PROMPT["2. Its text goes in the prompt<br/>as EVIDENCE, not orders"]

    TABLES --> SCHEMA[Schema tool]
    SCHEMA --> GEN
    PROMPT --> GEN

    GEN[Write SQL] --> VAL{"Validate<br/>parse, SELECT only"}
    VAL -->|rejected| GEN
    VAL -->|ok| EXEC[("Run read-only<br/>on Postgres")]
    EXEC -->|error, max 3 tries| GEN
    EXEC --> ANA[Compute numbers<br/>fixed pandas functions]
    ANA --> ANS[Write the answer]
    ANS --> OUT["Answer + SQL + which<br/>lessons were used"]

    style MEM fill:#e8f0fe
    style TOPK fill:#e6f4ea
    style OUT fill:#e6f4ea
    style DROP1 fill:#fce8e6
    style DROP2 fill:#fce8e6
    style DROP3 fill:#fce8e6
```

## Loop B — receiving a correction (write to memory)

```mermaid
flowchart TD
    OUT["Answer the agent gave"] --> HUMAN["Analyst types a correction<br/><i>'rank them by spend, not id'</i>"]

    HUMAN --> CLASS["Classifier fills a fixed form<br/>type · mistake · correction<br/>· <b>reusable lesson</b> · clarity<br/>· tables · claims"]

    CLASS --> KIND{"What kind of<br/>claim is it?"}

    KIND -->|"empirical<br/>'refunds rose in March'"| V1["Run SQL.<br/>Do the numbers agree?"]
    KIND -->|"causal<br/>'refunds CAUSED it'"| V2["Measure how much it explains.<br/>If it claims ALWAYS,<br/>hunt a counterexample"]
    KIND -->|"procedural<br/>'rank by spend'"| V3["Executable? Material?<br/>Conflicts with a stored rule?"]

    V1 --> STATUS{Status}
    V2 --> STATUS
    V3 --> STATUS

    STATUS -->|data agrees| VER["VERIFIED"]
    STATUS -->|data disagrees| REJ["REJECTED<br/>kept for audit, never used"]
    STATUS -->|clashes with existing| CON["CONFLICTING<br/>neither applied blindly"]
    STATUS -->|not checked yet| PEN["PENDING"]

    VER --> EMB2[Embed lesson + original question]
    PEN --> EMB2
    CON --> EMB2
    REJ --> EMB2

    EMB2 --> STORE[("feedback_memory<br/>lesson, embedding, status,<br/>confidence, provenance,<br/>tables, raw text")]

    STORE -.->|"read by Loop A<br/>on the next question"| NEXT["Future question"]

    style HUMAN fill:#fff4e5
    style VER fill:#e6f4ea
    style REJ fill:#fce8e6
    style CON fill:#fef7e0
    style STORE fill:#e8f0fe
```

## The three systems being compared

All three run the identical pipeline. Only the retrieval filter differs, which
is what makes the comparison meaningful — any score difference comes from the
memory, not from a prompt someone edited.

```mermaid
flowchart LR
    subgraph S1["1 · Baseline"]
        direction TB
        A1[Question] --> B1[No retrieval] --> C1[Answer]
    end

    subgraph S2["2 · Feedback RAG"]
        direction TB
        A2[Question] --> B2["Retrieve<br/>PENDING + VERIFIED"] --> C2[Answer]
        B2 -.->|"will happily use<br/>an unchecked lesson"| W2(("risk"))
    end

    subgraph S3["3 · Verified feedback"]
        direction TB
        A3[Question] --> B3["Retrieve VERIFIED only<br/>re-ranked by confidence"] --> C3[Answer]
    end

    style W2 fill:#fce8e6
    style S1 fill:#f8f9fa
    style S2 fill:#fff4e5
    style S3 fill:#e6f4ea
```

## What the experiment tests

A deliberately false lesson is injected: *"revenue declines are always caused
by refunds."*

- System 2 has no way to reject it, so it should adopt it and get worse.
- System 3 should reject it, because July 2026 is a month where revenue fell
  while refunds stayed flat — one counterexample is enough to refute a claim
  containing "always".

That gap between systems 2 and 3 is the result the project exists to produce.
