# Held-out analyst prior

A human hypothesis about relative greenwashing risk, recorded **before** scoring and
deliberately kept **out of** every prompt, weight, lexicon and config file in the pipeline.

Its only purpose is to check the pipeline **after** the fact. It is never an input.

## Why it is quarantined here

`/pipeline` contains no per-company branch, no company name in any prompt, and no
company-specific weight. You can verify that mechanically:

```bash
make check-leakage
```

If a hypothesis were encoded upstream, the pipeline would reproduce it by construction
and the resulting "signal" would carry zero information. Keeping it here means agreement
between prior and output is evidence *for* the method rather than an artefact of it.

## The prior

Stated by the analyst, 2026-09-19:

| Company | Prior expectation |
|---|---|
| H&M Group | highest greenwashing risk |
| Amazon | ambiguous / middle |
| Microsoft | most credible |

## Pipeline result (goalpost drift, Stage 6 — no LLM involved)

| Company | Republished cells | Restated | Median move | Max move | Drift score |
|---|---|---|---|---|---|
| H&M Group | 18 | 15 (83%) | 9.21% | 19.48% | **49.8** |
| Amazon | 24 | 5 (21%) | 1.90% | 2.58% | **15.7** |
| Microsoft | 0 | — | — | — | **unmeasurable** |

## Comparison

**The rank order of the two measurable companies matches the prior** (H&M > Amazon), and it
was produced by restatement arithmetic alone: how often a company republishes a prior year's
figure at a different value, and by how much. No language model, no lexicon, no tuning.

Two honest caveats that must be stated alongside that agreement:

1. **n = 2 measurable companies.** Reproducing a 2-item ordering is weak evidence. With
   n=2 there is a 50% chance of agreement under a null of random ordering. This is
   suggestive, not confirmatory.
2. **Microsoft is unmeasurable, not vindicated.** It scores `None` because no metric-year
   appears in more than one of its reports — there is nothing to restate against. The prior
   says "most credible"; the pipeline says "cannot tell". **These are different claims, and
   conflating them would be exactly the error this file exists to prevent.** Microsoft's
   absolute Scope 1/2/3 series lives in an Environmental Data Fact Sheet that is not in
   `data/raw/`; adding it is the single highest-value way to make this company assessable.

## Falsification note

The drift metric would have contradicted the prior if H&M's restatements had been small and
frequent (housekeeping) while Amazon's had been rare and enormous. The scoring function
weights rate and magnitude jointly (0.45 / 0.40) precisely so neither pattern alone dominates.
That the prior survived a metric capable of contradicting it is the part worth reporting.
