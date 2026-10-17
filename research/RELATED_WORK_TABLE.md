# MathGen — Related Work Table (Milestone 2)

Two parts:
- **A** compares the closest studies with MathGen's planned design, field by field.
- **B** is the full reference index. It is generated from the verified metadata, with venues only where confirmed on the official page.

Sources:
- **†** facts come from full-text reading. The verbatim quotes are in [`artifacts/m2_closest_papers_notes.md`](artifacts/m2_closest_papers_notes.md), or in `LITERATURE_REVIEW.md` §2.4 for the ones I re-checked myself.
- Rows marked *(abstract)* rely on the abstract only.
- "not stated" means the full text was searched and the item wasn't found. "—" means not applicable.

## A. Closest studies vs MathGen

| Study | Model, size, init | Task families | Tokenization varied? | PE / architecture varied? | Held-out design | Seen vs unseen measured? | Seeds / variance | Most relevant finding |
|---|---|---|---|---|---|---|---|---|
| **MathGen (planned; frozen in M3)** | decoder-only, ~5M, from scratch | 23 templates over 10 topics (integers, fractions, equations, f(x), 2-D vectors, 2×2 matrices, power-rule calculus, limits), trained jointly, JSON output | char / digit / bpe-num on one shared pre-tokenizer | RoPE, SwiGLU (one at a time) | instance-keyed hash split + dedup: unseen same-template, held-out value blocks, larger operands | **yes** | ≥3 seeds (5 for reference) | — |
| Baeumel et al. 2026† | 4 layers, d=256, ≈3.4M, scratch | 3-digit addition | **2×2 input × output** (digit vs whole number) | no | fixed 500-problem held-out set | no | **10 seeds/condition** | Effect is output-side; whole-number output converges unreliably |
| Zhou et al. 2024† | Pythia 70M–12B, scratch and fine-tuned | addition; multiplication (separate) | base 10 / 10² / 10³ token IDs | no | non-overlapping operand pairs; length OOD to 10¹⁶ | train ≈ eval compared | not stated | Base 10 most data-efficient from scratch; extrapolation ≈ 0 |
| Yang et al. 2024 (Number Cookbook)† | 0.1B / 0.9B / 3B scratch (+ a 1.3M PE probe); Llama fine-tuning | 17 numeric tasks × integer / float / fraction / scientific notation | 1/2/3-digit (+ random) | RoPE / NoPE / ALiBi; formats | ID vs longer-length OOD | no | 3 repeats for one PE table | One-digit best (gap closes at 3B); RoPE best ID, NoPE/ALiBi best length OOD |
| Kreitner et al. 2025† | nanoGPT-2, 117M, scratch | multi-task numeric mix | single-digit / triple-digit / xVal / FoNE / BitTokens | no | not checked | not checked | not checked | Single-digit beats subword among multi-token schemes |
| Lee et al. 2023† | NanoGPT 10.6M scratch; GPT-2; GPT-3 fine-tuned | +, −, ×, sin, sqrt; joint training | char default; char vs BPE in GPT-2 only | scale only | disjoint test; excluded operands; length | excluded-operand accuracy (Table 1) | not stated | Within-length generalization; no length generalization; scratchpads cut sample complexity |
| Kasliwal 2026† | 6 layers, d=384, ~10.6M non-embedding; WikiText-pretrained | +, −, ×, ÷ with scratchpads | GPT-2 BPE (fixed) | RoPE+RMSNorm+SwiGLU **bundled**; MoE | 4,000 held-out; 4-digit OOD | no | **single seed** | 0% on 4-digit OOD; dataloader-padding artifact |
| Satariyan et al. 2026 *(abstract)* | small Llama-style, scratch | 4×4 multiplication | separator token vs none | no | — | — | — | A space token between digits: 1% → 89% exact match |
| Charton 2021† | encoder–decoder ≤6 layers, 10–50M, scratch | 9 linear-algebra tasks (separate) | **P10 / P1000 / B1999 / FP15** | no | fresh random test sets; distribution OOD | no (argues against memorization) | not stated | P1000 dominates P10, the opposite of the integer results |
| Charton & Kempe 2024† | encoder–decoder 22–35M, scratch | GCD; mod-67 multiplication; eigenvalues | fixed (base 1000 / P1000) | no | fresh test sets; exclusion not stated | no | 5–30 inits | Repeated small sets beat single-use data |
| Saxton et al. 2019† | Transformer ~30M (+ LSTM/RMC), scratch | many modules: arithmetic, algebra, calculus, polynomials, … | char only | architecture families | ≤2% overlap bound (probabilistic); extrapolation sets | no | not stated | Interpolation 0.76 vs extrapolation 0.50 |
| Lample & Charton 2019† | 6 layers, d=512, scratch | integration; ODEs | prefix notation (fixed) | no | 5,000 held-out; cross-generator | no | not stated | 17.2% across generators |
| Nogueira et al. 2021† | T5 60M–3B fine-tuned; small from-scratch runs | addition; subtraction | 7 surface forms; bases; digit order | only in the vanilla model | interpolation/extrapolation by length; no overlap check | no | 5 data samples, 95% CI | Representation limits show only with little data |
| McLeish et al. 2024† | decoder 12–122M, scratch | +, −, ×, sorting | char (fixed), LSB-first | **Abacus / FIRE / NoPE / RoPE**; looping | length OOD | no | mean of 3, no error bars | 99% on 100-digit addition; "Rotary Embeddings are weak for length generalization" |
| Ye et al. 2024 (Physics 2.1)† | GPT2-small + RoPE, scratch | iGSM (one family, mod 23) | not stated | depth sweep | **template-level hash split**; op-length OOD | template-level | 2 seeds (depth only) | Not template memorization; depth matters |
| Singh & Strouse 2024† | GPT-3.5/4, inference only | 7–9-digit addition | L2R vs R2L chunking | — | — | — | none | R2L up to +20%; asks for the from-scratch "gold experiment" |
| Kazemnejad et al. 2023 *(abstract)* | decoder-only, scratch | reasoning + math tasks | no | APE / T5-rel / ALiBi / RoPE / NoPE | length OOD | no | not checked | NoPE best for length generalization |
| Narang et al. 2021† | encoder–decoder 223M | NLP only | fixed | many, incl. GLU variants and PE | standard | no | 5 runs (early loss) | SwiGLU helps; most modifications don't transfer |

**Reading the table:**
- No row other than MathGen combines a tokenization variable, seen-vs-unseen measurement, multiple seeds and multi-family training.
- Only Baeumel et al. matches our scale *and* reports seeds, and it covers a single task.
- See [`RESEARCH_GAP.md`](RESEARCH_GAP.md) §2 for the explicit check against the closest three.

## B. Full reference index

<!-- BEGIN GENERATED INDEX -->

*Generated by `research/lit_scripts/build_bib.py`; do not edit by hand. Grouped by each reference's primary theme; the last column lists all themes it informs.*

### T1 — Number representation and tokenization (22)

| Key | Reference | Title | Link | Venue (verified) | Themes |
|---|---|---|---|---|---|
| `sennrich2016bpe` | Sennrich et al. 2015 | Neural Machine Translation of Rare Words with Subword Units | [arXiv:1508.07909](https://arxiv.org/abs/1508.07909) | Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) | T1 |
| `wallace2019numbers` | Wallace et al. 2019 | Do NLP Models Know Numbers? Probing Numeracy in Embeddings | [arXiv:1909.07940](https://arxiv.org/abs/1909.07940) | Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) | T1 |
| `nogueira2021investigating` | Nogueira et al. 2021 | Investigating the Limitations of Transformers with Simple Arithmetic Tasks | [arXiv:2102.13019](https://arxiv.org/abs/2102.13019) | arXiv (venue not confirmed) | T1, T2 |
| `thawani2021representing` | Thawani et al. 2021 | Representing Numbers in NLP: a Survey and a Vision | [arXiv:2103.13136](https://arxiv.org/abs/2103.13136) | Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies | T1 |
| `charton2021linear` | Charton 2021 | Linear algebra with transformers | [arXiv:2112.01898](https://arxiv.org/abs/2112.01898) | Transactions on Machine Learning Research (TMLR) | T1, T5 |
| `chowdhery2022palm` | Chowdhery et al. 2022 | PaLM: Scaling Language Modeling with Pathways | [arXiv:2204.02311](https://arxiv.org/abs/2204.02311) | Journal of Machine Learning Research | T1 |
| `touvron2023llama` | Touvron et al. 2023 | LLaMA: Open and Efficient Foundation Language Models | [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) | arXiv (venue not confirmed) | T1 |
| `lee2023teaching` | Lee et al. 2023 | Teaching Arithmetic to Small Transformers | [arXiv:2307.03381](https://arxiv.org/abs/2307.03381) | International Conference on Learning Representations (ICLR 2024) | T1, T4, T5, T8 |
| `charton2024gcd` | Charton 2023 | Learning the greatest common divisor: explaining transformer predictions | [arXiv:2308.15594](https://arxiv.org/abs/2308.15594) | International Conference on Learning Representations (ICLR 2024) | T1, T5 |
| `golkar2023xval` | Golkar et al. 2023 | xVal: A Continuous Numerical Tokenization for Scientific Language Models | [arXiv:2310.02989](https://arxiv.org/abs/2310.02989) | arXiv (venue not confirmed) | T1 |
| `shen2023positional` | Shen et al. 2023 | Positional Description Matters for Transformers Arithmetic | [arXiv:2311.14737](https://arxiv.org/abs/2311.14737) | arXiv (venue not confirmed) | T1, T2 |
| `zhou2024robustly` | Zhou et al. 2024 | Transformers Can Achieve Length Generalization But Not Robustly | [arXiv:2402.09371](https://arxiv.org/abs/2402.09371) | arXiv (venue not confirmed) | T1, T2 |
| `singh2024tokenization` | Singh & Strouse 2024 | Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs | [arXiv:2402.14903](https://arxiv.org/abs/2402.14903) | arXiv (venue not confirmed) | T1 |
| `zhangli2024reverse` | Zhang-Li et al. 2024 | Reverse That Number! Decoding Order Matters in Arithmetic Learning | [arXiv:2403.05845](https://arxiv.org/abs/2403.05845) | arXiv (venue not confirmed) | T1, T8 |
| `mcleish2024abacus` | McLeish et al. 2024 | Transformers Can Do Arithmetic with the Right Embeddings | [arXiv:2405.17399](https://arxiv.org/abs/2405.17399) | Advances in Neural Information Processing Systems 37 (NeurIPS 2024) | T1, T2 |
| `zhou2024numeral` | Zhou et al. 2024 | Scaling Behavior for Large Language Models regarding Numeral Systems: An Example using Pythia | [arXiv:2409.17391](https://arxiv.org/abs/2409.17391) | Findings of the Association for Computational Linguistics: EMNLP 2024 | T1 |
| `levy2024digits` | Levy & Geva 2024 | Language Models Encode Numbers Using Digit Representations in Base 10 | [arXiv:2410.11781](https://arxiv.org/abs/2410.11781) | Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers) | T1 |
| `yang2024numbercookbook` | Yang et al. 2024 | Number Cookbook: Number Understanding of Language Models and How to Improve It | [arXiv:2411.03766](https://arxiv.org/abs/2411.03766) | International Conference on Learning Representations (ICLR 2025) | T1 |
| `kreitner2025bittokens` | Kreitner et al. 2025 | Efficient numeracy in language models through single-token number embeddings | [arXiv:2510.06824](https://arxiv.org/abs/2510.06824) | arXiv (venue not confirmed) | T1 |
| `baeumel2026output` | Baeumel et al. 2026 | When Tokenization is Secretly Output Supervision | [arXiv:2609.01386](https://arxiv.org/abs/2609.01386) | The 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP 2026), Main Conference | T1 |
| `satariyan2026separators` | Satariyan et al. 2026 | Separators Make Carry Propagation Learnable:The Geometry of Latent Carry in a Multiplication Transformer | [arXiv:2610.06605](https://arxiv.org/abs/2610.06605) | arXiv (venue not confirmed) | T1 |
| `huang2026nonbpe` | Huang et al. 2026 | Are Non-BPE Tokenizers Ready to Improve LLM Numeracy? | [official page](https://icml.cc/virtual/2026/82628) | 3rd AI for Math Workshop at ICML 2026 | T1 |

### T2 — Positional encoding, length and operand OOD generalization (7)

| Key | Reference | Title | Link | Venue (verified) | Themes |
|---|---|---|---|---|---|
| `su2021roformer` | Su et al. 2021 | RoFormer: Enhanced Transformer with Rotary Position Embedding | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | Neurocomputing | T2, T6 |
| `press2022alibi` | Press et al. 2021 | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation | [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) | International Conference on Learning Representations (ICLR 2022) | T2 |
| `anil2022length` | Anil et al. 2022 | Exploring Length Generalization in Large Language Models | [arXiv:2207.04901](https://arxiv.org/abs/2207.04901) | Advances in Neural Information Processing Systems 35 (NeurIPS 2022) | T2, T8 |
| `kazemnejad2023nope` | Kazemnejad et al. 2023 | The Impact of Positional Encoding on Length Generalization in Transformers | [arXiv:2305.19466](https://arxiv.org/abs/2305.19466) | Advances in Neural Information Processing Systems 36 (NeurIPS 2023) | T2, T6 |
| `jelassi2023length` | Jelassi et al. 2023 | Length Generalization in Arithmetic Transformers | [arXiv:2306.15400](https://arxiv.org/abs/2306.15400) | arXiv (venue not confirmed) | T2 |
| `zhou2023algorithms` | Zhou et al. 2023 | What Algorithms can Transformers Learn? A Study in Length Generalization | [arXiv:2310.16028](https://arxiv.org/abs/2310.16028) | International Conference on Learning Representations (ICLR 2024) | T2 |
| `singh2026relative` | Singh et al. 2026 | Relative Positions Generalize, Absolute Positions Memorize: An Implicit-Bias Account of Length Generalization in Attention | [arXiv:2607.18759](https://arxiv.org/abs/2607.18759) | arXiv (venue not confirmed) | T2 |

### T3 — Memorization vs generalization (25)

| Key | Reference | Title | Link | Venue (verified) | Themes |
|---|---|---|---|---|---|
| `zhang2017rethinking` | Zhang et al. 2016 | Understanding deep learning requires rethinking generalization | [arXiv:1611.03530](https://arxiv.org/abs/1611.03530) | International Conference on Learning Representations (ICLR 2017) | T3 |
| `arpit2017closer` | Arpit et al. 2017 | A Closer Look at Memorization in Deep Networks | [arXiv:1706.05394](https://arxiv.org/abs/1706.05394) | Proceedings of the 34th International Conference on Machine Learning (ICML 2017) | T3 |
| `feldman2020memorization` | Feldman 2019 | Does Learning Require Memorization? A Short Tale about a Long Tail | [arXiv:1906.05271](https://arxiv.org/abs/1906.05271) | Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing (STOC 2020) | T3 |
| `lee2022dedup` | Lee et al. 2021 | Deduplicating Training Data Makes Language Models Better | [arXiv:2107.06499](https://arxiv.org/abs/2107.06499) | Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) | T3, T7 |
| `zhang2023counterfactual` | Zhang et al. 2021 | Counterfactual Memorization in Neural Language Models | [arXiv:2112.12938](https://arxiv.org/abs/2112.12938) | Advances in Neural Information Processing Systems 36 (NeurIPS 2023) | T3 |
| `power2022grokking` | Power et al. 2022 | Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets | [arXiv:2201.02177](https://arxiv.org/abs/2201.02177) | arXiv (venue not confirmed) | T3, T4 |
| `carlini2023quantifying` | Carlini et al. 2022 | Quantifying Memorization Across Neural Language Models | [arXiv:2202.07646](https://arxiv.org/abs/2202.07646) | International Conference on Learning Representations (ICLR 2023) | T3 |
| `razeghi2022frequencies` | Razeghi et al. 2022 | Impact of Pretraining Term Frequencies on Few-Shot Reasoning | [arXiv:2202.07206](https://arxiv.org/abs/2202.07206) | Findings of the Association for Computational Linguistics: EMNLP 2022 | T3 |
| `magar2022contamination` | Magar & Schwartz 2022 | Data Contamination: From Memorization to Exploitation | [arXiv:2203.08242](https://arxiv.org/abs/2203.08242) | Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) | T3, T7 |
| `tirumala2022memorization` | Tirumala et al. 2022 | Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models | [arXiv:2205.10770](https://arxiv.org/abs/2205.10770) | Advances in Neural Information Processing Systems 35 (NeurIPS 2022) | T3 |
| `liu2022omnigrok` | Liu et al. 2022 | Omnigrok: Grokking Beyond Algorithmic Data | [arXiv:2210.01117](https://arxiv.org/abs/2210.01117) | International Conference on Learning Representations (ICLR 2023) | T3 |
| `nanda2023progress` | Nanda et al. 2023 | Progress measures for grokking via mechanistic interpretability | [arXiv:2301.05217](https://arxiv.org/abs/2301.05217) | International Conference on Learning Representations (ICLR 2023) | T3 |
| `dziri2023faith` | Dziri et al. 2023 | Faith and Fate: Limits of Transformers on Compositionality | [arXiv:2305.18654](https://arxiv.org/abs/2305.18654) | Advances in Neural Information Processing Systems 36 (NeurIPS 2023) | T3, T8 |
| `wu2024reasoning` | Wu et al. 2023 | Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks | [arXiv:2307.02477](https://arxiv.org/abs/2307.02477) | Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) | T3 |
| `maltoni2024arithmetic` | Maltoni & Ferrara 2023 | Arithmetic with Language Models: from Memorization to Computation | [arXiv:2308.01154](https://arxiv.org/abs/2308.01154) | Neural Networks | T3 |
| `doshi2023grok` | Doshi et al. 2023 | To grok or not to grok: Disentangling generalization and memorization on corrupted algorithmic datasets | [arXiv:2310.13061](https://arxiv.org/abs/2310.13061) | International Conference on Learning Representations (ICLR 2024) | T3 |
| `quirke2023addition` | Quirke & Barez 2023 | Understanding Addition in Transformers | [arXiv:2310.13121](https://arxiv.org/abs/2310.13121) | International Conference on Learning Representations (ICLR 2024) | T3 |
| `mirzadeh2024gsmsymbolic` | Mirzadeh et al. 2024 | GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models | [arXiv:2410.05229](https://arxiv.org/abs/2410.05229) | International Conference on Learning Representations (ICLR 2025) | T3, T7 |
| `deng2024symbolic` | Deng et al. 2024 | Language Models are Symbolic Learners in Arithmetic | [arXiv:2410.15580](https://arxiv.org/abs/2410.15580) | Transactions on Machine Learning Research (TMLR) | T3 |
| `nikankin2024heuristics` | Nikankin et al. 2024 | Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics | [arXiv:2410.21272](https://arxiv.org/abs/2410.21272) | International Conference on Learning Representations (ICLR 2025) | T3 |
| `barron2025toobig` | Barron & White 2025 | Too Big to Think: Capacity, Memorization, and Generalization in Pre-Trained Transformers | [arXiv:2506.09099](https://arxiv.org/abs/2506.09099) | Tiny Titans: The next wave of On-Device Learning for Foundational Models (TTODLer-FM), Workshop at ICML 2025 | T3 |
| `singhal2025layernorm` | Singhal & Kim 2025 | Impact of Layer Norm on Memorization and Generalization in Transformers | [arXiv:2511.10566](https://arxiv.org/abs/2511.10566) | Advances in Neural Information Processing Systems 38 (NeurIPS 2025) | T3, T6 |
| `harma2026compact` | Härmä et al. 2026 | On Memorization and Generalization in Compact Transformers | [doi:10.3390/electronics15091847](https://doi.org/10.3390/electronics15091847) | Electronics | T3 |
| `zhao2026shattered` | Zhao et al. 2026 | Shattered Compositionality: Counterintuitive Learning Dynamics of Transformers for Arithmetic | [arXiv:2601.22510](https://arxiv.org/abs/2601.22510) | arXiv (venue not confirmed) | T3, T8 |
| `so2026slower` | So et al. 2026 | Slower Generalization, Faster Memorization: A Sweet Spot in Algorithmic Learning | [arXiv:2605.14659](https://arxiv.org/abs/2605.14659) | arXiv (venue not confirmed) | T3, T4 |

### T4 — Data efficiency, repetition, phase transitions (8)

| Key | Reference | Title | Link | Venue (verified) | Themes |
|---|---|---|---|---|---|
| `kaplan2020scaling` | Kaplan et al. 2020 | Scaling Laws for Neural Language Models | [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) | arXiv (venue not confirmed) | T4, T6 |
| `hoffmann2022chinchilla` | Hoffmann et al. 2022 | Training Compute-Optimal Large Language Models | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) | Advances in Neural Information Processing Systems 35 (NeurIPS 2022) | T4 |
| `hernandez2022repeated` | Hernandez et al. 2022 | Scaling Laws and Interpretability of Learning from Repeated Data | [arXiv:2205.10487](https://arxiv.org/abs/2205.10487) | arXiv (venue not confirmed) | T4 |
| `eldan2023tinystories` | Eldan & Li 2023 | TinyStories: How Small Can Language Models Be and Still Speak Coherent English? | [arXiv:2305.07759](https://arxiv.org/abs/2305.07759) | arXiv (venue not confirmed) | T4, T6 |
| `muennighoff2023constrained` | Muennighoff et al. 2023 | Scaling Data-Constrained Language Models | [arXiv:2305.16264](https://arxiv.org/abs/2305.16264) | Advances in Neural Information Processing Systems 36 (NeurIPS 2023) | T4 |
| `zhu2024critical` | Zhu et al. 2024 | Critical Data Size of Language Models from a Grokking Perspective | [arXiv:2401.10463](https://arxiv.org/abs/2401.10463) | arXiv (venue not confirmed) | T4 |
| `charton2024repeated` | Charton & Kempe 2024 | Emergent properties with repeated examples | [arXiv:2410.07041](https://arxiv.org/abs/2410.07041) | arXiv (venue not confirmed) | T4, T3 |
| `kasliwal2026tiny` | Kasliwal 2026 | Algorithmic Scratchpads and Curriculum Staging for Arithmetic Reasoning in Tiny Transformers | [arXiv:2610.09003](https://arxiv.org/abs/2610.09003) | arXiv (venue not confirmed) | T4, T6, T8 |

### T5 — Structured math with sequence models (8)

| Key | Reference | Title | Link | Venue (verified) | Themes |
|---|---|---|---|---|---|
| `saxton2019analysing` | Saxton et al. 2019 | Analysing Mathematical Reasoning Abilities of Neural Models | [arXiv:1904.01557](https://arxiv.org/abs/1904.01557) | International Conference on Learning Representations (ICLR 2019) | T5, T7 |
| `lample2020symbolic` | Lample & Charton 2019 | Deep Learning for Symbolic Mathematics | [arXiv:1912.01412](https://arxiv.org/abs/1912.01412) | International Conference on Learning Representations (ICLR 2020) | T5 |
| `hendrycks2021math` | Hendrycks et al. 2021 | Measuring Mathematical Problem Solving With the MATH Dataset | [arXiv:2103.03874](https://arxiv.org/abs/2103.03874) | Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1 (NeurIPS Datasets and Benchmarks 2021) | T5 |
| `welleck2022brittleness` | Welleck et al. 2021 | Symbolic Brittleness in Sequence Models: on Systematic Generalization in Symbolic Mathematics | [arXiv:2109.13986](https://arxiv.org/abs/2109.13986) | Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-22) | T5, T7 |
| `cobbe2021gsm8k` | Cobbe et al. 2021 | Training Verifiers to Solve Math Word Problems | [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) | arXiv (venue not confirmed) | T5 |
| `charton2022doing` | Charton 2022 | What is my math transformer doing? -- Three results on interpretability and generalization | [arXiv:2211.00170](https://arxiv.org/abs/2211.00170) | arXiv (venue not confirmed) | T5, T3 |
| `liu2023tinygsm` | Liu et al. 2023 | TinyGSM: achieving >80% on GSM8k with small language models | [arXiv:2312.09241](https://arxiv.org/abs/2312.09241) | arXiv (venue not confirmed) | T5, T4 |
| `ye2024physics21` | Ye et al. 2024 | Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process | [arXiv:2407.20311](https://arxiv.org/abs/2407.20311) | International Conference on Learning Representations (ICLR 2025) | T5, T3 |

### T6 — Architecture components at small scale (6)

| Key | Reference | Title | Link | Venue (verified) | Themes |
|---|---|---|---|---|---|
| `vaswani2017attention` | Vaswani et al. 2017 | Attention Is All You Need | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Advances in Neural Information Processing Systems 30 (NIPS 2017) | T6 |
| `zhang2019rmsnorm` | Zhang & Sennrich 2019 | Root Mean Square Layer Normalization | [arXiv:1910.07467](https://arxiv.org/abs/1910.07467) | Advances in Neural Information Processing Systems 32 (NeurIPS 2019) | T6 |
| `shazeer2020glu` | Shazeer 2020 | GLU Variants Improve Transformer | [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) | arXiv (venue not confirmed) | T6 |
| `xiong2020layernorm` | Xiong et al. 2020 | On Layer Normalization in the Transformer Architecture | [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) | Proceedings of the 37th International Conference on Machine Learning (ICML 2020) | T6 |
| `narang2021modifications` | Narang et al. 2021 | Do Transformer Modifications Transfer Across Implementations and Applications? | [arXiv:2102.11972](https://arxiv.org/abs/2102.11972) | Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing | T6 |
| `smithline2026sparsity` | Smithline & Mascioli 2026 | Sparsity Moves Computation: How FFN Architecture Reshapes Attention in Small Transformers | [arXiv:2605.09403](https://arxiv.org/abs/2605.09403) | Mechanistic Interpretability Workshop at ICML 2026 (virtual poster) | T6 |

### T7 — Split design and generalization taxonomy (4)

| Key | Reference | Title | Link | Venue (verified) | Themes |
|---|---|---|---|---|---|
| `lake2018scan` | Lake & Baroni 2017 | Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks | [arXiv:1711.00350](https://arxiv.org/abs/1711.00350) | Proceedings of the 35th International Conference on Machine Learning (ICML 2018) | T7 |
| `hupkes2020compositionality` | Hupkes et al. 2019 | Compositionality decomposed: how do neural networks generalise? | [arXiv:1908.08351](https://arxiv.org/abs/1908.08351) | Journal of Artificial Intelligence Research | T7 |
| `keysers2020cfq` | Keysers et al. 2019 | Measuring Compositional Generalization: A Comprehensive Method on Realistic Data | [arXiv:1912.09713](https://arxiv.org/abs/1912.09713) | International Conference on Learning Representations (ICLR 2020) | T7 |
| `hupkes2023taxonomy` | Hupkes et al. 2022 | State-of-the-art generalisation research in NLP: A taxonomy and review | [arXiv:2210.03050](https://arxiv.org/abs/2210.03050) | Nature Machine Intelligence | T7 |

### T8 — Output ordering: scratchpads, hints, answer-first (4)

| Key | Reference | Title | Link | Venue (verified) | Themes |
|---|---|---|---|---|---|
| `nye2021scratchpad` | Nye et al. 2021 | Show Your Work: Scratchpads for Intermediate Computation with Language Models | [arXiv:2112.00114](https://arxiv.org/abs/2112.00114) | arXiv (venue not confirmed) | T8 |
| `wei2022cot` | Wei et al. 2022 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) | Advances in Neural Information Processing Systems 35 (NeurIPS 2022) | T8 |
| `chen2024posthinking` | Chen et al. 2024 | Distilling Reasoning Ability from Large Language Models with Adaptive Thinking | [arXiv:2404.09170](https://arxiv.org/abs/2404.09170) | IEEE Transactions on Neural Networks and Learning Systems | T8 |
| `he2026ceiling` | He et al. 2026 | Ceiling of a Task: When Can a Transformer Succeed Without Its Chain of Thought? | [arXiv:2609.33134](https://arxiv.org/abs/2609.33134) | arXiv (venue not confirmed) | T8 |

### T9 — Evaluation statistics and reproducibility (8)

| Key | Reference | Title | Link | Venue (verified) | Themes |
|---|---|---|---|---|---|
| `wilson1927probable` | Wilson 1927 | Probable Inference, the Law of Succession, and Statistical Inference | [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953) | Journal of the American Statistical Association | T9 |
| `dror2018hitchhiker` | Dror et al. 2018 | The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing | [doi:10.18653/v1/P18-1128](https://doi.org/10.18653/v1/P18-1128) | Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) | T9 |
| `dodge2019showyourwork` | Dodge et al. 2019 | Show Your Work: Improved Reporting of Experimental Results | [arXiv:1909.03004](https://arxiv.org/abs/1909.03004) | Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) | T9 |
| `pineau2021reproducibility` | Pineau et al. 2020 | Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program) | [arXiv:2003.12206](https://arxiv.org/abs/2003.12206) | Journal of Machine Learning Research | T9 |
| `bouthillier2021variance` | Bouthillier et al. 2021 | Accounting for Variance in Machine Learning Benchmarks | [arXiv:2103.03098](https://arxiv.org/abs/2103.03098) | Proceedings of Machine Learning and Systems 3 (MLSys 2021) | T9 |
| `agarwal2021precipice` | Agarwal et al. 2021 | Deep Reinforcement Learning at the Edge of the Statistical Precipice | [arXiv:2108.13264](https://arxiv.org/abs/2108.13264) | Advances in Neural Information Processing Systems 34 (NeurIPS 2021) | T9 |
| `picard2021seed` | Picard 2021 | Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision | [arXiv:2109.08203](https://arxiv.org/abs/2109.08203) | arXiv (venue not confirmed) | T9 |
| `madaan2024variance` | Madaan et al. 2024 | Quantifying Variance in Evaluation Benchmarks | [arXiv:2406.10229](https://arxiv.org/abs/2406.10229) | arXiv (venue not confirmed) | T9 |

<!-- END GENERATED INDEX -->
