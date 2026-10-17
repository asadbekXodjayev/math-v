> **Provenance.** Raw working notes from the M2 search agent (2026-10-17), committed unedited as evidence. Corrections made during verification are in `research/LITERATURE_REVIEW.md` §2.4. Its RQ numbering is RQ1 tokenization, RQ2 memorization, RQ3 data efficiency, RQ4 architecture, RQ5 output order. Machine-readable query log: `m2_novelty_query_log.jsonl`.

# M2 novelty-threat literature search (run 2026-10-17)

Planned study: ~5M-param decoder-only Transformers trained from scratch on template-generated structured math (~23 families: integer arithmetic, fractions, linear/quadratic equations, 2-D vectors, 2x2 matrices, power-rule derivatives/integrals, polynomial limits); question -> JSON (hints + answer).
RQ1 number tokenization (char vs digit-split vs multi-digit BPE, shared pre-tokenizer, matched non-embedding params/steps) · RQ2 memorization vs generalization (seen instances / unseen same-template / held-out operand blocks / out-of-range operands; instance-keyed hash splits + dedup) · RQ3 data efficiency / instance-space coverage · RQ4 RoPE vs learned-absolute vs NoPE; SwiGLU vs GELU · RQ5 hints-first vs answer-first. ≥3 seeds, CIs.

## Method and caveats
- Primary source: arXiv API (`http://export.arxiv.org/api/query`), 102 `search_query` variants, max_results=50 each, ~3.2 s between calls (script: `arxiv_q.py`; raw JSON per query in `arxiv_results/qNN.json`; machine log in `arxiv_results/query_log.jsonl`).
- Every paper in the candidate table was re-fetched with `id_list=` and its title/authors/abstract/comment/journal_ref taken from the API response (`ids_*.txt` files). arXiv IDs below are exactly as the API returned them.
- One venue paper without an arXiv ID found (ICML 2026, "Are Non-BPE Tokenizers Ready to Improve LLM Numeracy?") is confirmed only via its official icml.cc page.
- Full-text spot checks (arXiv HTML / ar5iv via WebFetch, which summarizes; treat fine details as "to double-check"): 2411.03766, 2409.17391, 2609.01386, 2510.06824, 2610.09003, 2307.03381.
- OpenAlex: attempted, HTTP 429 "Insufficient budget ... free daily budget ... used up" for this IP (no API key created, to avoid sending the user's email). Semantic Scholar not used (429 per instructions).
- arXiv `abs:` search is keyword/stem based and quirky (e.g. `abs:tokeniz` matches nothing; date-sorted broad queries q12/q63 return loosely matching noise). Hit counts are `opensearch:totalResults`.

## Query log — arXiv API (all run 2026-10-17)

| # | arXiv search_query | sort | date run | total hits (opensearch:totalResults) |
|---|---|---|---|---|
| q01 | `abs:"number tokenization" AND abs:arithmetic` | relevance | 2026-10-17 | 2 |
| q02 | `abs:tokenization AND abs:arithmetic AND abs:digit` | relevance | 2026-10-17 | 28 |
| q03 | `abs:"from scratch" AND abs:arithmetic AND abs:transformer` | relevance | 2026-10-17 | 14 |
| q04 | `abs:memorization AND abs:arithmetic` | relevance | 2026-10-17 | 62 |
| q05 | `abs:"length generalization" AND abs:positional AND abs:arithmetic` | relevance | 2026-10-17 | 9 |
| q06 | `abs:synthetic AND abs:mathematical AND abs:"small transformer"` | relevance | 2026-10-17 | 1 |
| q07 | `abs:digit AND abs:tokeniz AND abs:number` | relevance | 2026-10-17 | 0 |
| q08 | `abs:"linear algebra" AND abs:transformer AND abs:synthetic` | relevance | 2026-10-17 | 8 |
| q09 | `abs:"symbolic integration" AND abs:transformer` | relevance | 2026-10-17 | 7 |
| q10 | `abs:"repeated examples" AND abs:transformer` | relevance | 2026-10-17 | 1 |
| q11 | `abs:seen AND abs:unseen AND abs:arithmetic AND abs:transformer` | relevance | 2026-10-17 | 1 |
| q12 | `abs:arithmetic AND abs:tokenization` | submittedDate | 2026-10-17 | 316 |
| q13 | `abs:"small language models" AND abs:arithmetic AND abs:"from scratch"` | relevance | 2026-10-17 | 2 |
| q14 | `abs:scratchpad AND abs:arithmetic AND abs:"small transformer"` | relevance | 2026-10-17 | 0 |
| q15 | `abs:"positional encoding" AND abs:arithmetic AND abs:NoPE` | relevance | 2026-10-17 | 0 |
| q16 | `abs:"number representation" AND abs:arithmetic AND abs:transformer` | relevance | 2026-10-17 | 4 |
| q17 | `abs:tokenization AND abs:numbers AND abs:"from scratch"` | relevance | 2026-10-17 | 68 |
| q18 | `ti:tokenization AND ti:arithmetic` | relevance | 2026-10-17 | 5 |
| q19 | `ti:number AND ti:tokenization` | relevance | 2026-10-17 | 15 |
| q20 | `abs:"numeral systems" AND abs:language` | relevance | 2026-10-17 | 62 |
| q21 | `abs:"digit-level" AND abs:tokenization` | relevance | 2026-10-17 | 0 |
| q22 | `abs:"right-to-left" AND abs:tokenization` | relevance | 2026-10-17 | 25 |
| q23 | `abs:"small transformers" AND abs:arithmetic` | relevance | 2026-10-17 | 4 |
| q24 | `abs:synthetic AND abs:mathematics AND abs:"from scratch" AND abs:transformer` | relevance | 2026-10-17 | 3 |
| q25 | `abs:interpolation AND abs:extrapolation AND abs:arithmetic AND abs:transformer` | relevance | 2026-10-17 | 4 |
| q26 | `abs:"data efficiency" AND abs:arithmetic AND abs:transformer` | relevance | 2026-10-17 | 4 |
| q27 | `abs:scratchpad AND abs:arithmetic` | relevance | 2026-10-17 | 15 |
| q28 | `abs:"chain of thought" AND abs:"from scratch" AND abs:arithmetic` | relevance | 2026-10-17 | 2 |
| q29 | `abs:"positional encoding" AND abs:arithmetic` | relevance | 2026-10-17 | 16 |
| q30 | `abs:SwiGLU AND abs:arithmetic` | relevance | 2026-10-17 | 4 |
| q31 | `abs:"mathematical reasoning" AND abs:"small transformer"` | relevance | 2026-10-17 | 1 |
| q32 | `abs:memorization AND abs:generalization AND abs:synthetic AND abs:math AND abs:transformer` | relevance | 2026-10-17 | 2 |
| q33 | `abs:tokenization AND abs:memorization AND abs:numbers` | relevance | 2026-10-17 | 21 |
| q34 | `abs:"answer first" AND abs:reasoning` | relevance | 2026-10-17 | 31 |
| q35 | `abs:"linear equations" AND abs:transformer AND abs:trained` | relevance | 2026-10-17 | 5 |
| q36 | `abs:"symbolic mathematics" AND abs:transformer` | relevance | 2026-10-17 | 13 |
| q37 | `abs:"number formatting" AND abs:arithmetic` | relevance | 2026-10-17 | 19 |
| q38 | `abs:"tokenizer" AND abs:arithmetic AND abs:"small"` | relevance | 2026-10-17 | 47 |
| q39 | `ti:arithmetic AND ti:transformers` | submittedDate | 2026-10-17 | 53 |
| q40 | `ti:arithmetic AND ti:"language models"` | submittedDate | 2026-10-17 | 50 |
| q41 | `abs:"character-level" AND abs:arithmetic AND abs:transformer` | relevance | 2026-10-17 | 2 |
| q42 | `abs:"multi-digit" AND abs:tokens AND abs:arithmetic` | relevance | 2026-10-17 | 6 |
| q43 | `abs:"compositional generalization" AND abs:arithmetic AND abs:transformer` | relevance | 2026-10-17 | 7 |
| q44 | `abs:"tiny transformers" AND abs:arithmetic` | relevance | 2026-10-17 | 2 |
| q45 | `abs:"length generalization" AND abs:NoPE` | relevance | 2026-10-17 | 5 |
| q46 | `abs:calculus AND abs:transformer AND abs:synthetic` | relevance | 2026-10-17 | 6 |
| q47 | `abs:fractions AND abs:transformer AND abs:arithmetic` | relevance | 2026-10-17 | 48 |
| q48 | `abs:"before the answer" AND abs:"chain of thought" AND abs:training` | relevance | 2026-10-17 | 5 |
| q49 | `abs:"teaching arithmetic"` | relevance | 2026-10-17 | 2 |
| q50 | `abs:memorization AND abs:transformer AND abs:arithmetic` | submittedDate | 2026-10-17 | 23 |
| q51 | `abs:multiplication AND abs:"from scratch" AND abs:transformer` | relevance | 2026-10-17 | 141 |
| q52 | `abs:"mathematics dataset" AND abs:extrapolation` | relevance | 2026-10-17 | 1 |
| q53 | `abs:"held-out" AND abs:operands` | relevance | 2026-10-17 | 6 |
| q54 | `abs:tokenization AND abs:"mathematical reasoning" AND abs:digits` | relevance | 2026-10-17 | 7 |
| q55 | `abs:"number embeddings" AND abs:arithmetic` | relevance | 2026-10-17 | 2 |
| q56 | `abs:"matrix" AND abs:transformers AND abs:"linear algebra" AND abs:trained` | relevance | 2026-10-17 | 6 |
| q57 | `abs:"data diversity" AND abs:arithmetic` | relevance | 2026-10-17 | 1 |
| q58 | `abs:"positional encodings" AND abs:"small" AND abs:"algorithmic"` | relevance | 2026-10-17 | 8 |
| q59 | `abs:tokenization AND abs:"length generalization" AND abs:arithmetic` | relevance | 2026-10-17 | 2 |
| q60 | `abs:digit AND abs:tokenization AND abs:"from scratch"` | relevance | 2026-10-17 | 10 |
| q61 | `abs:"number tokenization"` | relevance | 2026-10-17 | 8 |
| q62 | `abs:"numerical tokenization"` | relevance | 2026-10-17 | 41 |
| q63 | `abs:tokenization AND abs:numbers AND abs:generalization AND abs:transformers` | submittedDate | 2026-10-17 | 359 |
| q64 | `abs:BPE AND abs:arithmetic AND abs:digits` | relevance | 2026-10-17 | 3 |
| q65 | `abs:mathematics AND abs:"trained from scratch" AND abs:algebra` | relevance | 2026-10-17 | 1 |
| q66 | `abs:"symbolic differentiation" AND abs:transformer` | relevance | 2026-10-17 | 4 |
| q67 | `abs:transformers AND abs:matrix AND abs:"eigenvalues" AND abs:trained` | relevance | 2026-10-17 | 36 |
| q68 | `abs:"synthetic math" AND abs:small AND abs:model` | relevance | 2026-10-17 | 1 |
| q69 | `abs:template AND abs:math AND abs:memorization AND abs:generalization` | relevance | 2026-10-17 | 1 |
| q70 | `abs:"chain of thought" AND abs:order AND abs:answer AND abs:"from scratch"` | relevance | 2026-10-17 | 1 |
| q71 | `abs:rationale AND abs:"after the answer"` | relevance | 2026-10-17 | 4 |
| q72 | `abs:"multi-task" AND abs:arithmetic AND abs:transformer AND abs:scratch` | relevance | 2026-10-17 | 1 |
| q73 | `abs:"Physics of Language Models"` | relevance | 2026-10-17 | 2 |
| q74 | `abs:"out-of-distribution" AND abs:operands AND abs:transformer` | relevance | 2026-10-17 | 1 |
| q75 | `abs:interpolation AND abs:extrapolation AND abs:mathematics AND abs:neural` | relevance | 2026-10-17 | 13 |
| q76 | `abs:numeracy AND abs:"from scratch"` | relevance | 2026-10-17 | 2 |
| q77 | `ti:numeracy` | submittedDate | 2026-10-17 | 20 |
| q78 | `abs:"grade school" AND abs:synthetic AND abs:"from scratch" AND abs:pretrain` | relevance | 2026-10-17 | 0 |
| q79 | `ti:"Non-BPE" AND ti:numeracy` | relevance | 2026-10-17 | 0 |
| q80 | `abs:numeracy AND abs:tokenizer` | submittedDate | 2026-10-17 | 7 |
| q81 | `abs:"digit chunking"` | relevance | 2026-10-17 | 1 |
| q82 | `abs:"answer" AND abs:"chain-of-thought" AND abs:order AND abs:arithmetic AND abs:transformers` | relevance | 2026-10-17 | 1 |
| q83 | `abs:scratchpad AND abs:"from scratch"` | submittedDate | 2026-10-17 | 3 |
| q84 | `abs:"sample complexity" AND abs:arithmetic AND abs:transformer` | relevance | 2026-10-17 | 2 |
| q85 | `abs:"in-distribution" AND abs:"out-of-distribution" AND abs:arithmetic AND abs:"from scratch"` | relevance | 2026-10-17 | 1 |
| q86 | `abs:polynomial AND abs:transformer AND abs:"from scratch" AND abs:symbolic` | relevance | 2026-10-17 | 0 |
| q87 | `abs:"symbolic integration"` | submittedDate | 2026-10-17 | 104 |
| q88 | `abs:architecture AND abs:ablation AND abs:arithmetic AND abs:"small transformers"` | relevance | 2026-10-17 | 1 |
| q89 | `abs:template AND abs:synthetic AND abs:arithmetic AND abs:generalization AND abs:transformer` | relevance | 2026-10-17 | 0 |
| q90 | `abs:"memorization" AND abs:"generalization" AND abs:"tokenization"` | relevance | 2026-10-17 | 184 |
| q91 | `abs:"training set" AND abs:"test set" AND abs:arithmetic AND abs:"memorize"` | relevance | 2026-10-17 | 0 |
| q92 | `abs:"multiple tasks" AND abs:arithmetic AND abs:"small" AND abs:transformers AND abs:trained` | relevance | 2026-10-17 | 0 |
| q93 | `abs:"critical data size"` | relevance | 2026-10-17 | 5 |
| q94 | `abs:arithmetic AND abs:tokenization AND abs:"small language models"` | relevance | 2026-10-17 | 5 |
| q95 | `abs:"number representation" AND abs:"language models" AND abs:"from scratch"` | relevance | 2026-10-17 | 1 |
| q96 | `abs:"post-thinking"` | relevance | 2026-10-17 | 5 |
| q97 | `ti:"length generalization"` | submittedDate | 2026-10-17 | 68 |
| q98 | `abs:"equation solving" AND abs:transformer AND abs:synthetic` | relevance | 2026-10-17 | 0 |
| q99 | `abs:tokenization AND abs:"from scratch" AND abs:arithmetic` | relevance | 2026-10-17 | 14 |
| q100 | `abs:tokenizer AND abs:"from scratch" AND abs:math` | relevance | 2026-10-17 | 25 |
| q101 | `abs:hints AND abs:arithmetic AND abs:transformer AND abs:answer` | relevance | 2026-10-17 | 0 |
| q102 | `abs:"intermediate steps" AND abs:"from scratch" AND abs:transformer AND abs:arithmetic` | relevance | 2026-10-17 | 1 |

## Query log — WebSearch / venue pages (2026-10-17)

| # | query | result count | new leads |
|---|---|---|---|
| w1 | number tokenization arithmetic small transformers trained from scratch memorization generalization 2025 | 9 links | 2604.13082, 2406.00075, ACL-Findings-2025 precision paper (2410.13857) |
| w2 | ICLR 2025 OR NeurIPS 2025 tokenization digits arithmetic "from scratch" single-digit vs multi-digit tokens length generalization | 9 | confirmed Number Cookbook tokenizer experiments |
| w3 | small transformer trained from scratch synthetic math templates equations derivatives matrices memorization vs generalization unseen instances 2026 arXiv | 9 | 2605.14659; MDPI Electronics 2026 "On Memorization and Generalization in Compact Transformers" (not arXiv, not verified further) |
| w4 | "tokenization" arithmetic "held-out" operands "memorization" small language model from scratch digit-level vs BPE ... EMNLP/ACL 2025 | 10 | ICML 2026 "Are Non-BPE Tokenizers Ready to Improve LLM Numeracy?" |
| w5 | "Are Non-BPE Tokenizers Ready to Improve LLM Numeracy" arXiv | 9 | no arXiv version found |
| w6 | answer-first versus chain-of-thought-first output order training small transformer from scratch arithmetic | 9 | 2601.22035 (LLM-scale, not pursued) |
| w7 | tokenizer comparison numbers digit-level vs BPE "trained from scratch" math reasoning small language models 2025 2026 | 9 | 2404.14395 Paramanu-Ganita (math-domain BPE, no digit comparison) |
| w8 | transformer trained from scratch synthetic algebra calculus problems linear/quadratic equations derivatives ... interpolation extrapolation | 9 | nothing closer than Charton 2021 |
| w9 | RoPE vs NoPE vs absolute PE, SwiGLU vs GELU ablation small transformer arithmetic from scratch seeds CIs 2025 | 9 | no SwiGLU-vs-GELU arithmetic study found |
| w10-12 | venue checks: Saxton (ICLR 2019 poster page), McLeish (NeurIPS 2024 proceedings), Charton & Kempe (not found) | — | — |
| o1 | OpenAlex `works?search=` (4 queries) | HTTP 429 | budget exhausted, skipped |

## Candidate list (most relevant; ordered by novelty threat)

Threat = how much it pre-empts our *combined* study (and, in brackets, the single RQ it most threatens).

| arXiv ID | First author et al. | Year | Title (exact, from API) | Venue (source) | RQs | What they did / found (short verbatim abstract quote) | Threat |
|---|---|---|---|---|---|---|---|
| 2411.03766v3 | Haotong Yang et al. | 2024 | Number Cookbook: Number Understanding of Language Models and How to Improve It | ICLR 2025 poster (arXiv comment) | RQ1, RQ4, (RQ5) | Benchmark of 17 numeric tasks x 4 representations (incl. fractions, sci-notation); "we train small models with existing and potential techniques for enhancing NUPA (such as tokenizers, PEs, and number formats)". Full text: 1/2/3-digit tokenizers at 0.1B-3B (one-digit best for length generalization); RoPE vs NoPE vs Alibi at 100M; one task per model; CoT only on fine-tuned Llama-3.1-8B. | HIGH for RQ1/RQ4 in isolation; medium overall — no algebra/calculus/matrix families, no seen-vs-unseen memorization split, ≥100M params |
| 2409.17391v2 | Zhejian Zhou et al. | 2024 | Scaling Behavior for Large Language Models regarding Numeral Systems: An Example using Pythia | EMNLP 2024 Findings (arXiv comment) | RQ1, RQ3, (RQ2) | Base-10 vs base-100/1000 number tokenization: base 10 is "consistently more data-efficient than a base $10^{2}$ or $10^{3}$ system across training data scale, model sizes under from-scratch training settings". Full text: Pythia 70M-12B, add/mult only, OOD = longer operands, train-vs-eval overfitting check for addition. | HIGH for RQ1xRQ3 on integer arithmetic; medium overall — only 2 ops, ≥70M, no char-level arm, no held-out blocks |
| 2609.01386v1 | Tanja Baeumel et al. | 2026 | When Tokenization is Secretly Output Supervision | EMNLP 2026 Main (arXiv comment) | RQ1 | Decouples input vs output number tokenization in a controlled numeric task: "differences in task performance, training dynamics, and model internals are induced by output tokenization and largely invariant to input tokenization". Full text (via fetch): from scratch, 4 layers d=256, digit vs 3-digit tokens, addition only, 10 seeds. | MEDIUM-HIGH for RQ1 at our scale (single task); methodologically important: our RQ1 conflates input+output tokenization unless we discuss/decouple it |
| 2610.09003v1 | Sourabh Kasliwal | 2026 | Algorithmic Scratchpads and Curriculum Staging for Arithmetic Reasoning in Tiny Transformers | preprint (Oct 6 2026) | RQ4, RQ5, RQ2 (OOD) | "(~10.6M non-embedding parameters, 49.3M total)" trained on +,-,x,/ scratchpads; "modern architectural primitives (RoPE, RMSNorm, SwiGLU) ... substantially improve additive reasoning over baseline GPT-2"; "performance collapses to 0.00% on unseen 4-digit operands". Full text: GPT-2 BPE only, WikiText pre-training required, bundled ablation, single seed, no overlap/dedup check. | MEDIUM — closest in scale and spirit, but no tokenization comparison, no memorization measurement, not from scratch, 1 seed |
| 2307.03381v1 | Nayoung Lee et al. | 2023 (older) | Teaching Arithmetic to Small Transformers | ICLR 2024 (iclr.cc proceedings) | RQ3, RQ5, (RQ2) | NanoGPT ~10.6M from random init, char-level: "simple formatting changes can significantly improve accuracy. This leads to sharp phase transitions as a function of training data scale"; plain vs reverse vs scratchpad formats; deliberate exclusion of numbers/digits from training. | HIGH for RQ3/RQ5 on arithmetic (the canonical prior); low for RQ1/RQ4 and multi-family |
| 2510.06824v2 | Linus Kreitner et al. | 2025 | Efficient numeracy in language models through single-token number embeddings | preprint | RQ1 | Proposes BitTokens; "BitTokens allow even small language models to learn algorithms that solve basic arithmetic operations nearly perfectly". Full text: nanoGPT-2 from scratch, compares single-digit, triple-digit subword, xVal, FoNE, BitTokens on a multi-task numeric mix + FineWeb text. | MEDIUM for RQ1 (multi-task, from scratch) — no algebraic families, no memorization splits |
| 1904.01557v1 | David Saxton et al. | 2019 (older) | Analysing Mathematical Reasoning Abilities of Neural Models | ICLR 2019 (iclr.cc poster page) | multi-family, RQ2, (RQ4) | Template-generated suite "covering arithmetic, algebra, probability and calculus, enables the construction of training and test splits designed to clearly illuminate the capabilities and failure-modes of different architectures"; interpolation vs extrapolation test sets; LSTM vs Transformer. | MEDIUM (older very close match for "multi-family synthetic math + generalization splits") — char-level only, no tokenization/memorization-of-seen-instances analysis, large models |
| 2305.19466v2 | Amirhossein Kazemnejad et al. | 2023 (older) | The Impact of Positional Encoding on Length Generalization in Transformers | NeurIPS 2023 (arXiv comment) | RQ4, RQ5 | Small decoder-only from scratch, APE/T5-rel/ALiBi/Rotary/NoPE on reasoning+math tasks: "NoPE outperforms other explicit positional encoding methods"; "scratchpad is not always helpful ... and its format highly impacts the model's performance". | MEDIUM for RQ4 (length generalization, not in-distribution/memorization) |
| 2506.09099v2 | Joshua Barron et al. | 2025 | Too Big to Think: Capacity, Memorization, and Generalization in Pre-Trained Transformers | ICML 2025 Tiny Titans workshop (arXiv comment) | RQ2 | Capacity-limited char-level Transformers from scratch: "small models extrapolate to unseen arithmetic cases but fail to memorize facts, while larger models memorize but fail to extrapolate". | MEDIUM-LOW — "memorization" = factual recall task, not seen math instances; no tokenization |
| 2410.07041v1 | François Charton et al. | 2024 | Emergent properties with repeated examples | not verified | RQ3, RQ2 | GCD, modular multiplication, matrix eigenvalues: "models trained on smaller sets of repeated examples outperform models trained on larger sets of single-use examples". | MEDIUM-LOW for RQ3 (repetition vs diversity is a confound we must control/report) |
| 2604.27551v1 | Henrik Voigt et al. | 2026 | Beyond the Training Distribution: Mapping Generalization Boundaries in Neural Program Synthesis | preprint | RQ2, RQ3 | "To rigorously assess whether models are truly generalizing or merely retrieving memorized templates, we introduce a strictly controlled program synthesis environment based on a domain-specific arithmetic grammar"; support extrapolation drops >30%. | LOW-MEDIUM — split methodology analogue, different task (program synthesis) |
| 2402.14903v1 | Aaditya K. Singh et al. | 2024 | Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs | preprint | RQ1 | "right-to-left tokenization (enforced by comma separating numbers at inference time) leads to largely improved performance" (GPT-3.5/4). | LOW (frontier LLMs, inference-time only) — must cite; motivates L2R vs R2L chunking choice for our multi-digit BPE arm |
| 2610.06605v1 | Sama Satariyan et al. | 2026 | Separators Make Carry Propagation Learnable:The Geometry of Latent Carry in a Multiplication Transformer | preprint | RQ1 (format), RQ5 | Small Llama-style from scratch, 4x4 multiplication, no CoT: "inserting a space token between digits raises exact-match accuracy from 1% to 89%". | LOW-MEDIUM — single task, but shows input-format/separator choice alone can dominate; our shared pre-tokenizer must control this |
| 2102.13019v3 | Rodrigo Nogueira et al. | 2021 (older) | Investigating the Limitations of Transformers with Simple Arithmetic Tasks | not verified | RQ1 | Surface form matters: "the model fails to learn addition of five-digit numbers when using subwords (e.g., "32"), and it struggles to learn with character-level representations (e.g., "3 2")". | LOW (pretrained T5, add/sub) — classic citation for RQ1 |
| 2112.01898v2 | François Charton | 2021 (older) | Linear algebra with transformers | TMLR 2022 (arXiv comment) | RQ1 (number encodings), matrices, RQ2 (OOD) | Nine matrix problems; "introduce and discuss four encoding schemes to represent real numbers"; models "can generalize out of their training distribution". | LOW-MEDIUM (older) — matrix family + encoding comparison, but no BPE/char comparison, no memorization split |
| 2405.17399v2 | Sean McLeish et al. | 2024 | Transformers Can Do Arithmetic with the Right Embeddings | NeurIPS 2024 (proceedings page) | RQ4, RQ2 (OOD length) | Abacus digit-position embeddings: "training on only 20 digit numbers ... achieving up to 99% accuracy on 100 digit addition problems". | LOW — length-generalization focus |
| 2402.09371v1 | Yongchao Zhou et al. | 2024 | Transformers Can Achieve Length Generalization But Not Robustly | preprint | RQ4, methodology | Format x PE determines length generalization, which is fragile, "leading to large variances across different random seeds". | LOW — supports our ≥3-seed + CI design |
| 2605.09403v2 | Gabriel Smithline et al. | 2026 | Sparsity Moves Computation: How FFN Architecture Reshapes Attention in Small Transformers | preprint | RQ4 (FFN) | One-layer Transformers on digit addition/modular arithmetic: "Comparing dense FFNs, gated linear units (GLUs), mixture-of-experts (MoE), and MoE-GLUs"; GLU gating "rotates task-relevant Fourier structure out of the per-neuron basis". | LOW — mechanistic, not accuracy/generalization; only GLU-vs-dense prior on arithmetic found |
| 2601.22510v2 | Xingyu Zhao et al. | 2026 | Shattered Compositionality: Counterintuitive Learning Dynamics of Transformers for Arithmetic | preprint | RQ2, RQ5 | Synthetic arithmetic: "correlational matching to the training data, rather than causal or procedural composition, shapes learning dynamics"; not fixed by "pure model scaling or scratchpad supervision". | LOW-MEDIUM |
| 2609.33134v1 | Jiashu He et al. | 2026 | Ceiling of a Task: When Can a Transformer Succeed Without Its Chain of Thought? | preprint | RQ5 | Theory + small from-scratch transformers on finite-group word problems: "chain-trained models solve every input length and fall to chance when the chain is erased". | LOW-MEDIUM for RQ5 (gives a theoretical lens: our families are mostly shallow, so answer-first may hit no "ceiling") |
| 2404.09170v6 | Xiaoshu Chen et al. | 2024 | Distilling Reasoning Ability from Large Language Models with Adaptive Thinking | IEEE TNNLS (arXiv comment) | RQ5 | Answer-first vs rationale-first for fine-tuned SLMs: "we propose a robust post-thinking mechanism to generate answers before rationale". | LOW-MEDIUM — the only direct answer-first-vs-rationale-first comparison found, but on pretrained SLMs and NL tasks |
| 2407.20311v1 | Tian Ye et al. | 2024 | Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process | ICML 2024 tutorial video (arXiv comment) | RQ2 | Synthetic GSM-style data, from-scratch GPT-2-size models: "Can language models truly develop reasoning skills, or do they simply memorize templates?" | LOW — word problems, no tokenization axis |
| (no arXiv ID found) | Yizhan Huang et al. | 2026 | Are Non-BPE Tokenizers Ready to Improve LLM Numeracy? | ICML 2026 (icml.cc/virtual/2026/82628; OpenReview 1yWx6FfWEP) | RQ1 | Digit chunking via cross-tokenizer distillation, number encodings, byte-level LLMs: "all three current alternatives could not outperform BPE in the arithmetic-NL Pareto frontier". | LOW — LLM scale with NL trade-off; from-scratch status unclear |

Background / cite-if-relevant (verified via API, lower relevance): 1912.01412v1 Lample & Charton, "Deep Learning for Symbolic Mathematics" (2019; integration/ODEs, seq2seq from scratch); 2308.01154v4 Maltoni & Ferrara, "Arithmetic with Language Models: from Memorization to Computation" (Neural Networks 179, 2024; binary add/mult, light LM); 2401.10463v3 Zhu et al., "Critical Data Size of Language Models from a Grokking Perspective" (2024; data-size phase transition memorization->generalization); 2605.14659v1 So et al., "Slower Generalization, Faster Memorization: A Sweet Spot in Algorithmic Learning" (2026; non-monotone data-size effect on structured outputs); 2607.18759v1 Singh et al., "Relative Positions Generalize, Absolute Positions Memorize: An Implicit-Bias Account of Length Generalization in Attention" (2026); 2410.15580v2 Deng et al., "Language Models are Symbolic Learners in Arithmetic" (TMLR 2026); 2604.13082v2 Gomezjurado Gonzalez, "The Long Delay to Arithmetic Generalization: When Learned Representations Outrun Behavior" (2026; numeral base as inductive bias, Collatz).

## Assessment

No paper found answers the combined question (number tokenization x seen/unseen/held-out/out-of-range splits x ~23 structured-math families x ~5M from-scratch, multi-seed). Each single RQ, however, has strong prior work on *integer arithmetic*:
- RQ1 alone on integer arithmetic is NOT novel (2409.17391, 2411.03766, 2609.01386, 2510.06824; consistent finding: single-digit > multi-digit tokens, esp. for data efficiency / extrapolation). 2609.01386 additionally shows the effect is driven by *output* tokenization.
- RQ4 PE comparisons are well covered for length generalization (2305.19466, 2411.03766); SwiGLU-vs-GELU on math generalization: only mechanistic/bundled evidence (2605.09403, 2610.09003).
- RQ3/RQ5 on arithmetic: Lee et al. 2307.03381 is the canonical prior; direct answer-first vs hints-first at small scale from scratch was not found.

Closest three:
1. Number Cookbook (2411.03766, ICLR 2025) — tokenizer (1/2/3-digit) + PE (RoPE/NoPE/Alibi) + number-format ablations from scratch over several numeric representations including fractions. Does not cover: equations/vectors/matrices/calculus/limits; seen-vs-unseen instance memorization or held-out operand blocks; ~5M scale (smallest 0.1B; one model per task); SwiGLU vs GELU; hints-first vs answer-first for small from-scratch models; seed counts not stated for from-scratch runs.
2. Zhou et al. (2409.17391, EMNLP 2024 Findings) — tokenization granularity (base 10/100/1000) x data scale x model size, from scratch, OOD longer operands. Does not cover: char-level arm or shared pre-tokenizer at matched non-embedding params; anything beyond add/mult; held-out operand blocks / unseen-same-template splits (only an overfitting check); architecture or output-order ablations; models below 70M.
3. Kasliwal (2610.09003, Oct 2026 preprint) — tiny (~10.6M non-emb) arithmetic transformers with scratchpads, RoPE/RMSNorm/SwiGLU, unseen 4-digit operands, Wilson CIs. Does not cover: tokenization comparison (GPT-2 BPE only); training purely from scratch (requires WikiText pre-training); memorization measurement / train-test dedup; isolated per-component ablations; multiple seeds; families beyond the 4 basic ops.
(Runner-up: Baeumel et al. 2609.01386 — tiny from-scratch, 10 seeds, input/output tokenization decoupled, but addition only and no memorization splits.)

Remaining novelty (defensible claims): (a) breadth — RQ1/RQ4/RQ5 effects measured across ~23 structured-math families, not just integer arithmetic; (b) instance-keyed hash splits with dedup giving a clean seen / unseen-same-template / held-out-block / out-of-range ladder, crossed with tokenization; (c) ~5M scale with matched non-embedding params and >=3 seeds + CIs; (d) hints-first vs answer-first at small from-scratch scale. Recommend citing and positioning against 2411.03766, 2409.17391, 2609.01386, 2610.09003, 2307.03381, 1904.01557, and explicitly addressing the input-vs-output tokenization confound raised by 2609.01386 and the separator/format confound shown by 2610.06605.
