# BDH Driven Continuous Narrative Reasoning

**Team:** BDH_ON_TOP  
**Authors:** Piyush Panda, Priyam Ghosh, Priyansh Rawat  

## 📖 Project Overview

In this project, we address the complex challenge of narrative consistency detection. In our setting, we process data samples consisting of a natural-language claim paired with a specific entity identifier. 

Our primary objective is to predict whether the given claim is `consistent` with the entity's accumulated narrative state, or if it will `contradict` it. 

**Our Central/Core Idea:** Instead of relying on a single monolithic sequence model to internally store all prior facts, which we found often struggles with dilution over long contexts, we externalize persistence into an explicit, structured memory window. We then treat the classification task as a similarity decision between our representation of the claim and the memory trace we learned for the referenced entity.

---

## 🧠 Core Features & How We Built Them

### 1. Structured Memory (`CausalMemory`)
**What we built:** We maintain a persistent memory matrix $M \in \mathbb{R}^{N \times D}$ inside a custom memory module. We update this memory using a gated write/retain rule. Given a current slot vector $m$ and a contextualized fact vector $\tilde{f}$, we compute an input gate $i$ and a forget gate $f$. We then update the memory sequentially:

$$m^{\prime} = f \cdot m + i \cdot \tilde{f}$$

We also apply a reconstruction head to map the memory back into the embedding space, producing a comparable memory trace:

$$r = Rec(m) \in \mathbb{R}^{D}$$

**Our Motivation:** For long narratives, we noticed that attempting to use a single pooled embedding for the entire content acts as a lossy bottleneck. We found that salient early facts get diluted, causing the model to place an unnecessary and unintended bias on late-arriving information. By compressing durable information into a structured, persistent memory over time, we severely limit dilution from irrelevant background text.

### 2. Sequential Ingestion with Positional Context
**What we built:** As we sequentially ingest a novel word-by-word, we embed each word into a vector $f_t = Enc(w_t) \in \mathbb{R}^D$. To provide context, we inject two learned positional projections: a time embedding (tracking the word index within a segment) and a segment embedding (tracking the chapter index). The context-aware embedding becomes:

$$\overline{f}_t = f_t \odot (t_t + l_t)$$

**Our Motivation:** With this approach, we significantly reduce destructive interference across long text streams. It ensures that repeated tokens behave differently depending on when and in which segment they occur, greatly improving the stability of our recurrent memory updates.

### 3. Dual-Objective Training (Distinguishing Signal from Noise)
**What we built:** We distinguish causal signals from stylistic noise through a hybrid training loss:

* **Unsupervised Ingestion:** We update the memory sequentially while optimizing a reconstruction objective. We apply auxiliary gate regularizers to encourage conservative writes and stable retention.
  $$\mathcal{L}_{recon} = ||Rec(m^{\prime}) - f_t||_2^2$$
* **Supervised Consistency:** Using labeled pairs, we compute a statement embedding $b$ and retrieve an entity trace $r$, scoring them using cosine similarity $s = cos(r,b)$. We train with a hinge-style margin loss that forces high similarity for `consistent` samples and applies a negative margin for `contradict` samples.

**Our Motivation:** Much of a long narrative is incidental filler. Our unsupervised ingestion stabilizes the mechanics of the memory under long streams, while our supervised phase provides the task signal, ensuring our memory representations only reinforce facts that improve label prediction.

### 4. Hashed Pseudo-Tokenization & Mean-Pooling
**What we built:** We map each whitespace-delimited word to a pseudo-token ID using hash-based indexing and embed it with a learned lookup table. We represent a full statement by mean-pooling these word embeddings:

$$b = \frac{1}{|\mathcal{W}|} \sum_{w \in \mathcal{W}} Enc(w)$$

**Our Motivation:** This implementation serves as a lightweight surrogate for a true tokenizer. We intentionally trade semantic fidelity for processing speed and absolute simplicity, allowing us to iterate rapidly and clearly observe our failure modes.

---

## ⚠️ Key Limitations & Known Failure Cases

Because we intentionally prioritize simplicity and controllability in our implementation, we accept several known trade-offs:

1. **Hash-Based Indexing Collisions:** Our hashing directly induces bucket collisions, causing interference between entirely unrelated tokens or entities.
2. **Syntax & Scope Collapse:** Our bag-of-words mean pooling discards word order, negation scope, and compositional semantics, yielding failures on negations and role reversals (e.g., confusing "X killed Y" with "Y killed X").
3. **Catastrophic Drift:** Repeated writes during long or repetitive contexts can cause our memory to drift toward recent narrative text rather than overarching facts.
4. **Spurious Similarity (Lexical Bias):** Our cosine similarity over averaged embeddings is highly sensitive to shared vocabulary. Contradictions with high lexical overlap (e.g., "alive" vs "dead") can be misclassified as consistent.
5. **Tokenization Artifacts:** Our whitespace splitting is brittle under punctuation and OCR noise. Variants like `John`, `John,`, and `john` behave as completely different tokens.
6. **Weak Co-Reference Handling:** Without explicit co-reference resolution, evidence we process via pronouns may fail to align with the correct entity memory trace.

---

## 📁 Local Setup & Reproducibility

### Expected Project Layout
```text
/BDH_ON_TOP_KDSH_2026:
├── main.py
├── setup.ps1
├── requirements.txt
├── /checkpoints
│   └── best_bdh_model.pth
├── /outputs
    └── results.csv
