# Sequential Instruction Tuning of Phi-3.5 Mini with Teacher-Generated JSON Supervision

## Overview

This project implements the full two-stage sequential instruction-tuning pipeline with judge evaluation. The student model is **`microsoft/Phi-3.5-mini-instruct`**, trained first on Alpaca-style general instruction data and then further tuned on a teacher-generated structured JSON dataset. The main goal was to study whether Stage 2 JSON specialization improves structured-output behavior without causing catastrophic forgetting of the general instruction-following ability learned in Stage 1.

The final outcome is that **catastrophic forgetting was not observed**. Automatic Alpaca metrics changed only slightly from Checkpoint 1 to Checkpoint 2, and judge comparisons frequently preferred Checkpoint 2 over Checkpoint 1 on held-out Alpaca prompts. At the same time, JSON validity became very high, although strict schema-based JSON gains remained modest under the benchmark design used here.

---

## 1. Methodology

### Student Model
The student model used in this project was **`microsoft/Phi-3.5-mini-instruct`**. This model was the first choice as it is recommended in the case study guidelines as a practical small-model default for QLoRA-based post-training. It is large enough to support meaningful instruction-following behavior, yet small enough to fine-tune efficiently in a realistic research workflow.

### Stage 1: Alpaca Fine-Tuning
For Stage 1, I used **`yahma/alpaca-cleaned`** as the Alpaca-style instruction dataset. This cleaned version was selected because it is better curated and easier to justify than the original raw Alpaca release. To keep the project computationally manageable while still preserving experimental value, I trained on a **10,000-example subset** rather than the full dataset.

The Alpaca held-out evaluation set contained **100 prompts** general instruction-following evaluation examples.

### Stage 2: Teacher-Generated JSON Dataset
For Stage 2, I built a structured JSON benchmark covering all five required task families:

- JSON extraction
- schema-constrained generation
- exact-label classification
- JSON repair
- tool-call argument generation

The prompts were written in a human-authored style and spread across diverse realistic topical domains, including:

- sports
- AI trends
- news
- crypto
- oil prices
- food
- environment
- technology
- entertainment
- domestic robots
- AI-driven mental health
- autonomous commuting
- healthcare operations
- education tools
- travel planning

The final JSON dataset size was:

- **300 train prompts**
- **200 eval prompts**

Each required task family received equal coverage:

- **60 training prompts per task type**
- **40 evaluation prompts per task type**

### Teacher Model
The teacher model used for synthetic supervision was **`llama-3.3-70b-instruct-awq`**. Each structured prompt was submitted to the teacher model, and the resulting outputs were validated for JSON correctness before inclusion in the final Stage 2 training set.

The final generation pass produced:

- **300 valid teacher-generated training examples**
- **200 held-out JSON evaluation prompts**
- **0 invalid teacher outputs**

This Stage 2 data construction process follows an imitation-learning / synthetic-data pipeline rather than classical soft-label distillation.

### Fine-Tuning Pipeline
The pipeline followed the required sequential structure:

- **Checkpoint 0**: base untuned `Phi-3.5-mini-instruct`
- **Checkpoint 1**: after Stage 1 Alpaca tuning
- **Checkpoint 2**: after Stage 2 JSON tuning

Training settings were:

- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- max sequence length: `2048`
- Stage 1 learning rate: `2e-5`
- Stage 2 learning rate: `1e-5`
- batch size: `2`
- gradient accumulation steps: `8`
- epochs: `2`

Stage 1 output:
- `artifacts/checkpoints/stage1_10k`

Stage 2 output:
- `artifacts/checkpoints/stage2_10k_300json`

### Infrastructure Note
Training was completed on **UTSA ARC HPC**. However, during inference, Phi-3.5 exhibited runtime compatibility problems in the original ARC environment, producing degenerate outputs. To obtain meaningful final predictions, the inference/evaluation phase was moved to an **alternate compatible GPU runtime**. This separation is documented here because it materially affected the engineering workflow and reproducibility considerations.

---

## 2. Experimental Setup

### Evaluation Sets
Each checkpoint was evaluated on the same fixed held-out sets:

- **Alpaca eval**: `100` prompts
- **JSON eval**: `200` prompts

This fixed-eval design is critical for the forgetting analysis because it allows direct comparison of model states without changing the benchmark.

### Automatic Metrics
For Alpaca-style evaluation, I computed:

- overlap-style F1 proxy
- ROUGE-L
- BERTScore F1
- task completion rate
- average output length

For JSON evaluation, I computed:

- JSON validity rate
- schema compliance rate
- exact match
- field-level F1
- error taxonomy

### Judge Evaluation
I also ran pairwise LLM-as-a-Judge comparisons using a structured judge schema that included:

- `prompt_id`
- `checkpoint_a`
- `checkpoint_b`
- `response_a_scores`
- `response_b_scores`
- `winner`
- `justification`

The judge compared all required checkpoint pairs for both Alpaca and JSON:

- Checkpoint 0 vs Checkpoint 1
- Checkpoint 1 vs Checkpoint 2
- Checkpoint 0 vs Checkpoint 2

---

## 3. Automatic Results

### Three-Checkpoint Comparison

| Checkpoint | Alpaca Overlap F1 | ROUGE-L | BERTScore F1 | JSON Validity | Schema Compliance | Exact Match | Field-Level F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Checkpoint 0 | 0.2606 | 0.2283 | 0.8672 | 0.985 | 0.360 | 0.000 | 0.000 |
| Checkpoint 1 | 0.2592 | 0.2267 | 0.8667 | 0.985 | 0.370 | 0.000 | 0.000 |
| Checkpoint 2 | 0.2551 | 0.2261 | 0.8669 | 0.985 | 0.425 | 0.000 | 0.000 |

### Interpretation
The Alpaca automatic metrics are very stable across checkpoints. Checkpoint 2 shows a small decrease in overlap F1 and ROUGE-L compared with Checkpoint 1, but the difference is very small. BERTScore slightly increased from Checkpoint 1 to Checkpoint 2.

On the JSON side, validity is consistently very high across checkpoints. The most notable gain is in **schema compliance**, which increases from `0.37` at Checkpoint 1 to `0.425` at Checkpoint 2. Exact match and field-level F1 remain zero, mainly because the benchmark references were coarse placeholder-style targets and did not always align with the richer JSON structures generated by the model.

---

## 4. Judge Results

### Alpaca Judge Results

| Comparison | Checkpoint A Win Rate | Checkpoint B Win Rate | Tie Rate |
| --- | ---: | ---: | ---: |
| Checkpoint 0 vs Checkpoint 1 | 0.27 | 0.29 | 0.44 |
| Checkpoint 1 vs Checkpoint 2 | 0.12 | 0.29 | 0.59 |
| Checkpoint 0 vs Checkpoint 2 | 0.19 | 0.29 | 0.52 |

### JSON Judge Results

| Comparison | Checkpoint A Win Rate | Checkpoint B Win Rate | Tie Rate |
| --- | ---: | ---: | ---: |
| Checkpoint 0 vs Checkpoint 1 | 0.16 | 0.02 | 0.82 |
| Checkpoint 1 vs Checkpoint 2 | 0.17 | 0.015 | 0.815 |
| Checkpoint 0 vs Checkpoint 2 | 0.175 | 0.025 | 0.80 |

### Interpretation
The Alpaca judge results are the strongest evidence in the project that Stage 2 did **not** catastrophically damage general instruction-following quality. In the direct Checkpoint 1 vs Checkpoint 2 comparison, Checkpoint 2 is preferred more than twice as often as Checkpoint 1 (`29%` vs `12%`), with the remaining cases mostly ties. This indicates that the Stage 2-tuned model retained general instruction capability and may even have slightly improved in judge-perceived clarity or completeness.

The JSON judge results were less favorable to the tuned checkpoints than expected. This likely reflects the mismatch between the benchmark’s simple reference schema and the model’s richer or differently structured JSON responses. As a result, the JSON judge results should be interpreted cautiously and in conjunction with the automatic validity/schema-compliance metrics.

---

## 5. Forgetting Analysis

The core research question of the assignment is whether continued fine-tuning on the JSON dataset causes catastrophic forgetting of the general instruction-following ability learned during Stage 1.

### Automatic Metric Change from Checkpoint 1 to Checkpoint 2

Using the final Alpaca metrics:

- Overlap F1 change: **-0.00413**
- ROUGE-L change: **-0.00054**
- BERTScore F1 change: **+0.00019**
- Average output length change: **-1.04 tokens**

### Interpretation
These changes are very small. The overlap F1 and ROUGE-L declines are minor, while BERTScore is slightly higher at Checkpoint 2. Taken together, the automatic metrics indicate **minor fluctuation, not catastrophic forgetting**.

This conclusion is reinforced by the Alpaca judge comparison:

- Checkpoint 1 wins: `12%`
- Checkpoint 2 wins: `29%`
- ties: `59%`

Thus, the strongest overall conclusion is that **Stage 2 did not induce catastrophic forgetting**. If anything, Checkpoint 2 retained or slightly improved general instruction-following quality in judge-based comparison, despite very small changes in automatic metrics.

---

## 6. Ablation Study

### Ablation: Stage 2 Dataset Size
As the required ablation, I compared a smaller Stage 2 synthetic JSON dataset against the final larger Stage 2 dataset.

#### Smaller Stage 2 JSON Set
- Alpaca overlap F1: `0.2591`
- ROUGE-L: `0.2266`
- BERTScore F1: `0.8661`
- JSON validity: `0.985`
- schema compliance: `0.36`

#### Larger Stage 2 JSON Set
- Alpaca overlap F1: `0.2551`
- ROUGE-L: `0.2261`
- BERTScore F1: `0.8669`
- JSON validity: `0.985`
- schema compliance: `0.425`

### Ablation Interpretation
Increasing the Stage 2 dataset size did **not** materially change JSON validity, which stayed at `0.985` in both settings. However, schema compliance improved from `0.36` to `0.425`, suggesting that the larger Stage 2 dataset helped the model better align with the expected structure. On the Alpaca side, the larger dataset caused only a very small decline in overlap F1 and ROUGE-L while BERTScore slightly improved.

This ablation supports the broader project conclusion: increasing Stage 2 JSON supervision did not cause catastrophic forgetting, while providing some improvement in structured-output adherence under the chosen benchmark.

---

## 7. Qualitative Example

A representative example where Checkpoint 2 outperformed Checkpoint 1 is prompt **`alpaca_47217`**:

**Prompt:**  
“Explain how to make a triangle out of three sticks?”

**Checkpoint 1 prediction:**  
It explains the problem using triangle inequality and begins a step-by-step answer, but the response is somewhat incomplete and less polished.

**Checkpoint 2 prediction:**  
It gives a clearer and slightly more structured explanation of how triangle formation depends on the triangle inequality theorem, with improved completeness and flow.

**Judge justification:**  
“Response B is more complete and clearer in its explanation, with a more accurate description of the triangle inequality theorem and its application to forming a triangle with three sticks.”

This example is useful because neither output is perfect, but the judge still preferred Checkpoint 2 for clarity and completeness. That aligns with the overall finding that Stage 2 tuning did not erase the model’s general reasoning ability.

---

## 8. Discussion

### Why JSON Exact Match Stayed Low
The JSON outputs were often semantically reasonable and structurally meaningful, but strict exact-match metrics remained at zero. There are two main reasons:

1. The model frequently produced richer or differently shaped JSON than the placeholder-style reference targets.
2. The benchmark references for several JSON tasks were simplified, so semantically good outputs were still penalized under exact-match evaluation.

Because of this, **JSON validity** and **schema compliance** were more informative than exact match or field-level F1 in this experiment.

### Runtime and Engineering Lessons
A major practical lesson from this project was that small-model post-training is not just about dataset and optimizer choices; runtime compatibility also matters. Phi-3.5 training ran successfully on ARC, but the original ARC inference runtime produced degenerate outputs, forcing the final inference/evaluation stage to be moved to an alternate compatible GPU runtime. This did not invalidate the training pipeline, but it did require careful engineering and reproducibility notes.

---

## 9. Conclusion

This project implemented a complete two-stage sequential instruction-tuning pipeline for a small language model using:

- `Phi-3.5-mini-instruct` as the student
- `yahma/alpaca-cleaned` for Stage 1
- teacher-generated JSON supervision from `llama-3.3-70b-instruct-awq` for Stage 2

The final result is that **catastrophic forgetting was not observed**. Automatic Alpaca metrics changed only slightly from Checkpoint 1 to Checkpoint 2, and judge evaluation often preferred the Stage 2 model over the Stage 1 model on held-out Alpaca prompts. Meanwhile, JSON validity was consistently high, and schema compliance improved in the final larger Stage 2 setup.

Overall, the experiment suggests that sequential fine-tuning can preserve general instruction-following ability while adding structured-output specialization, but benchmark design strongly affects how clearly those gains appear in quantitative metrics.

---

## Appendix: Prompting Notes

### Teacher Generation Prompting
The teacher-generation workflow used structured prompts that explicitly specified:
- task type
- expected schema
- realistic domain context
- requirement to return valid JSON

### Inference Prompting
For JSON inference, prompts were tightened to require:
- valid JSON only
- no markdown fences
- no explanatory text

This change was important because earlier runs often returned fenced JSON or explanatory prose, which hurt automatic JSON parsing.

### Judge Prompting
The judge prompt required:
- structured output
- per-dimension scoring
- explicit checkpoint identifiers
- a winner field
- a justification field

This made the judge outputs reproducible and easy to aggregate.

---

## Short README / Blog Summary

This project studies whether a small model can be fine-tuned sequentially for structured JSON behavior without forgetting earlier general instruction-following ability. I used `Phi-3.5-mini-instruct` as the student, trained it first on a `10,000`-example subset of `yahma/alpaca-cleaned`, and then continued tuning it on a teacher-generated JSON dataset created with `llama-3.3-70b-instruct-awq`.

The final results show **no evidence of catastrophic forgetting**. Automatic Alpaca metrics changed only slightly from Checkpoint 1 to Checkpoint 2, while judge comparisons often preferred the Stage 2 model over the Stage 1 model on held-out Alpaca prompts. JSON validity was very high, and schema compliance improved with the larger Stage 2 dataset, although strict exact-match metrics remained weak due to benchmark/reference mismatch.
