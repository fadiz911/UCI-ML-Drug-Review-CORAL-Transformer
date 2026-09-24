# UCI Machine Learning Drug Review: Dataset Executive Summary & Modeling Playbook

The UCI Machine Learning Drug Review dataset consists of patient-submitted narratives detailing their experiences with specific medications. The training partition contains **161,297 rows**, while the testing partition holds **53,766 rows**. Each record includes the drug name, the clinical condition it was prescribed for, the raw review text, a 10-star patient rating, the date, and a "useful count" indicating how many other users found the review helpful. Based on the exploratory data analysis, here is exactly what you are dealing with and how to navigate it in your preprocessing and modeling pipelines:

---

## 1. Critical Data Anomalies & Empirical Findings

- **Massive Cross-Partition Leakage:** Nearly **60% (32,131 rows)** of your test set contains exact duplicate review texts found in the training set, likely due to cross-posting on medical forums. If ignored, your model will simply memorize the training data and report artificially inflated accuracy.
- **Intra-Set Label Contradictions:** There are **71 unique review texts (affecting 380 rows)** where users submitted the exact same text but assigned contradictory ratings (e.g., a 9 and a 10 for the same narrative).
- **Hierarchical Condition Structure (Option B Parsing):** 
  - Total Processed Reviews: **159,498**
  - **95.4% (152,232 rows)** represent single primary conditions.
  - **4.6% (7,266 rows)** contain clinical inversions/sub-types formatted as `<Primary Condition>, <Sub-Type>` (e.g., `Diabetes, Type 2`, `Constipation, Chronic`, `Asthma, Maintenance`, `Breast Cancer, Metastatic`).
  - **Top Primary Condition Groups:** `Birth Control` (28,788), `Depression` (9,069), `Pain` (6,145), `Anxiety` (5,904), `Acne` (5,588), `Bipolar Disorder` (4,224), `Insomnia` (3,700), `Weight Loss` (3,609), `Obesity` (3,568), `ADHD` (3,383), `Diabetes` (2,694), `Emergency Contraception` (2,463), `Constipation` (2,449), `High Blood Pressure` (2,321), `Vaginal Yeast Infection` (2,274).
- **Severe Class Imbalance:** The 10-star ratings are bimodal, meaning patients mostly leave reviews when they either love (10 stars) or hate (1 star) a drug. When mapped to the 3-class problem, **66.3%** of the data becomes "Positive", leaving "Neutral" as a tiny **8.9% minority class**.
- **Uniform Text Lengths:** The median review is **84 words**, and review length does not change based on sentiment. A highly positive review and a highly negative review are structurally the same length, meaning your model must rely entirely on semantic meaning rather than word count.

---

## 2. Preprocessing Protocol

- **Review Quotation & Text Sanitization:** Raw text strings are wrapped in outer enclosing quotation marks (`"..."`). You must strip leading/trailing quotes (`"`, `'`, `“`, `”`), decode HTML entities, remove HTML tags, and strip recurring web-scraping metadata phrases (`"users found this comment helpful"`).
- **Hierarchical Condition Decomposition (Option B):** Avoid row explosion (do NOT split by comma into new rows), as duplicating reviews introduces severe data leakage and skews evaluation metrics. Instead, decompose into:
  - `condition_primary`: High-level disease category (e.g., `"Diabetes"`).
  - `condition_subtype`: Clinical qualifier (e.g., `"Type 2"`).
  - `condition_clean`: Natural clinical name (e.g., `"Type 2 Diabetes"`).
- **Sequence Truncation:** Because **99% of the reviews translate to 200 subword tokens or fewer**, set your LLM or deep learning context window / max length to **256 tokens**. This captures almost all text while keeping memory overhead minimal.

---

## 3. Modeling & Implementation Strategy

- **Condition-Conditioned NLP & LLM Prompting:** Prepend `condition_clean` or `condition_primary` to the review text during fine-tuning (e.g., `"[CONDITION] Type 2 Diabetes [REVIEW] ..."` or `"Patient treated for Type 2 Diabetes reports: ..."`). Side effect tolerances vary drastically across conditions (e.g., nausea is expected in chemotherapy but unacceptable in acne treatment); conditioning reviews on the primary condition allows sentiment models to learn condition-specific clinical baselines.
- **Categorical Feature Embeddings for Tabular / Hybrid Models:** For GBDT models (XGBoost, CatBoost, LightGBM) or multimodal architectures, pass `condition_primary` as a categorical feature (using Target Encoding or Categorical Embeddings) alongside TF-IDF/embeddings and `log_usefulCount`.
- **Validation Splitting:** Use **Grouped K-Fold Cross-Validation** (grouped by `review` text and stratified by `condition_primary`) to prevent review duplicate leakage across folds while maintaining balanced condition distributions.
- **Loss Function Engineering:** Implement **Class Weighting** or **Focal Loss** to prevent over-indexing on the 66% positive majority class, and use **Label Smoothing ($\epsilon = 0.1$)** to mitigate noisy/contradictory label gradients.
- **Evaluation Metrics:** Use **Quadratic Weighted Kappa (QWK)** for 10-class star rating regression/classification and **Macro-Averaged F1-Score** for 3-class target prediction (`rating_3_class`).
- **Explainability Candidates:** Use the **380 contradictory rows** for GenAI interpretability and error analysis, prompting LLMs to explain why identical review narratives received divergent patient rating scores.
