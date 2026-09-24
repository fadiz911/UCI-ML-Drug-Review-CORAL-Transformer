# 🏛️ CORAL Transformer Model Architecture & Evaluation Report

**Project:** Clinical NLP Drug Review Rating Prediction  
**Framework:** Consistent Rank Logits (CORAL) + Transformer Backbone (`distilbert-base-uncased`)  
**Target:** 10-Class Ordinal Ratings ($1 \le y \le 10$)  

---

## 📋 1. Executive Summary

This document provides a comprehensive technical reference for the neural network architecture, mathematical formulation, training methodology, and empirical evaluation results for predicting patient drug ratings from review text.

Unlike standard multiclass classification (which ignores the natural ordering of ratings) or standard regression (which assumes equal distance between classes and lacks calibrated probabilities), this project employs **Consistent Rank Logits (CORAL)**. CORAL reformulates the 10-class ordinal regression problem into **9 rank-consistent binary classification sub-tasks**, guaranteeing monotonic rank predictions while sharing feature representations across all rating thresholds.

---

## 🏗️ 2. Core Architecture & Workflow Diagram

```
                     ┌──────────────────────────────┐
                     │     Raw Review Text (x)      │
                     └──────────────┬───────────────┘
                                    │
                         (AutoTokenizer: max_len=256)
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   input_ids (B × 256)                             attention_mask (B × 256)
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    │
                   ┌────────────────▼────────────────┐
                   │  DistilBERT Transformer Engine  │
                   │    (6 Layers, 768 Hidden Dim)   │
                   └────────────────┬────────────────┘
                                    │
                       [CLS] Token Extraction
                                    │
                            Dropout (p = 0.2)
                                    │
                              h(x) ∈ ℝ^768
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
  Shared Linear Weight (768 → 1)             9 Ordinal Rank Biases
        W ∈ ℝ^(768×1)                             b_1, b_2, ..., b_9
               │                                         │
         g(x) = W^T h(x)                                 │
               │                                         │
               └────────────────────┬────────────────────┘
                                    │
                      z_k(x) = g(x) + b_k  for k ∈ {1..9}
                                    │
                             Logits: z ∈ ℝ^9
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
  [ Training Phase: CoralLoss ]                   [ Inference / Evaluation ]
   Binary Cross-Entropy over                      Discrete: r = 1 + ∑ 𝕀(z_k > 0)
     9 ordinal task labels                        Soft:     r_soft = 1 + ∑ σ(z_k)
```

---

## 🧮 3. Mathematical Formulation (CORAL)

### 3.1 Ordinal Binary Target Encoding
For a 10-class rating $y \in \{1, 2, \dots, 10\}$, CORAL constructs $K-1 = 9$ binary indicator target variables $y_k \in \{0, 1\}$:

$$y_k = \mathbb{I}(y > k) \quad \text{for } k \in \{1, 2, \dots, 9\}$$

#### Binary Label Map:
| Ground Truth Rating | $y_1$ ($>1$) | $y_2$ ($>2$) | $y_3$ ($>3$) | $y_4$ ($>4$) | $y_5$ ($>5$) | $y_6$ ($>6$) | $y_7$ ($>7$) | $y_8$ ($>8$) | $y_9$ ($>9$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **2** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **3** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **5** | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| **10** | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

*Note: Rating 3 is represented by $y_1=1, y_2=1$ and $y_k=0$ for $k \ge 3$. Rating 3 is a label, not an input feature.*

### 3.2 Rank Logit Projection
Let $h(x) \in \mathbb{R}^{768}$ be the contextual embedding extracted from the `[CLS]` token after dropout.
CORAL computes task logits $z_k(x)$ using a **single weight vector** $W \in \mathbb{R}^{768 \times 1}$ and **9 independent rank bias terms** $b_1, b_2, \dots, b_9$:

$$g(x) = W^T h(x)$$
$$z_k(x) = g(x) + b_k \quad \text{for } k = 1, \dots, 9$$

*Key Advantage:* Because all tasks share the weight vector $W$, the model evaluates feature importance consistently across all ordinal levels, preventing threshold rank boundary crossing.

### 3.3 CORAL Loss Function
The model optimizes Binary Cross-Entropy with Logits averaged across the 9 task dimensions:

$$\mathcal{L}_{\text{CORAL}}(x, y) = \frac{1}{9} \sum_{k=1}^{9} \text{BCEWithLogits}\big(z_k(x), y_k\big)$$

$$\text{where } \text{BCEWithLogits}(z_k, y_k) = - \left[ y_k \log \sigma(z_k) + (1 - y_k) \log (1 - \sigma(z_k)) \right]$$

---

## 🔮 4. Rating Decoding Mechanics

At inference time, the model predicts final ordinal ratings from the 9 output logits $z(x) \in \mathbb{R}^9$:

### 4.1 Discrete Integer Rating ($\hat{r} \in \{1, \dots, 10\}$)
Calculated by summing the indicator functions of positive logits:

$$\hat{r} = 1 + \sum_{k=1}^{9} \mathbb{I}\big(z_k(x) > 0\big)$$

### 4.2 Continuous Soft Rating ($\hat{r}_{\text{soft}} \in [1.0, 10.0]$)
Calculated by summing the sigmoid probabilities across all binary tasks:

$$\hat{r}_{\text{soft}} = 1.0 + \sum_{k=1}^{9} \sigma\big(z_k(x)\big)$$

---

## ⚙️ 5. Implementation & Training Parameters

| Component / Hyperparameter | Configuration / Value | Description |
| :--- | :--- | :--- |
| **Backbone Model** | `distilbert-base-uncased` | Pretrained 6-layer Transformer |
| **Max Sequence Length** | `256` tokens | Truncation & padding limit for GPU VRAM optimization |
| **Batch Size** | `32` | Optimizer batch size |
| **Optimizer** | `AdamW` | Weight decay $0.01$ |
| **Backbone LR** | `2e-5` | Fine-tuning learning rate for DistilBERT layers |
| **Head LR** | `1e-3` | Higher learning rate for initial CORAL head weights |
| **Precision** | `AMP FP16` | Automatic Mixed Precision on CUDA |
| **Data Leakage Prevention** | GroupKFold / Deduplication | Grouped on review text to keep duplicate text out of val/test |

---

## 📊 6. Evaluation Results Summary

Evaluation performed on `cleaned_test.csv` (53,200 test samples):

### 6.1 Ordinal Summary Metrics
* **Total Samples:** `53,200`
* **Inference Speed:** `515.0 samples/sec` (103.29s total)
* **Discrete Mean Absolute Error (MAE):** **`0.5999`** *(Average error is ~0.6 rating points)*
* **Discrete Root Mean Squared Error (RMSE):** **`1.2106`**
* **Exact Match Accuracy:** **`61.91%`**
* **Off-by-1 Accuracy ($\le 1$ rating error):** **`88.70%`**
* **Kendall's Tau ($\tau$ rank correlation):** **`0.8179`**
* **Spearman's Rho ($\rho$ rank correlation):** **`0.8940`**

### 6.2 Per-Class Breakdown (Multiclass Report)

| Class Rating | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.7970 | 0.8220 | 0.8093 | 7,230 |
| **2** | 0.3869 | 0.3622 | 0.3741 | 2,308 |
| **3** | 0.4056 | 0.3963 | 0.4009 | 2,185 |
| **4** | 0.3516 | 0.3578 | 0.3547 | 1,632 |
| **5** | 0.4773 | 0.4843 | 0.4808 | 2,672 |
| **6** | 0.3872 | 0.3517 | 0.3686 | 2,090 |
| **7** | 0.4424 | 0.3969 | 0.4184 | 3,056 |
| **8** | 0.5213 | 0.5081 | 0.5146 | 6,087 |
| **9** | 0.5146 | 0.5336 | 0.5239 | 9,072 |
| **10** | 0.7915 | 0.8021 | 0.7967 | 16,868 |
| **Accuracy** | | | **61.91%** | 53,200 |
| **Macro Avg** | 0.5075 | 0.5015 | 0.5042 | 53,200 |
| **Weighted Avg** | 0.6155 | 0.6191 | 0.6171 | 53,200 |

---

## 🎯 7. Insights & Final Project Takeaways

1. **High Ordinal Rank Correlation:** Spearman's $\rho = 0.8940$ and Kendall's $\tau = 0.8179$ confirm that the model captures the underlying ordering of patient sentiment extremely well.
2. **Strong Off-by-1 Accuracy:** Over **88.7%** of predictions land within 1 rating point of the true rating (e.g. predicting 9 when true rating is 10 or 8).
3. **Extreme Class Performance:** Ratings 1 and 10 have the highest support and highest F1-scores (~0.80-0.81), representing strongly negative or strongly positive patient feedback. Intermediate ratings (2-7) exhibit moderate confusion due to subtle linguistic differences in mild/moderate sentiment.
