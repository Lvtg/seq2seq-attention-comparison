# Error Taxonomy

| Tag | Meaning | Report use |
| --- | --- | --- |
| rare word | Source contains low-frequency content words and models tend to omit or replace them. | Discuss vocabulary/data sparsity. |
| action error | Main action is mistranslated, weakened, or replaced. | Discuss semantic faithfulness. |
| modifier loss | Color, size, age, clothing, or other modifiers are dropped or altered. | Discuss detail retention. |
| repetition | A hypothesis repeats words or local phrases. | Discuss decoding/model fluency failure. |
| long sentence compression | Long source descriptions become shorter and lose clauses. | Discuss length sensitivity. |
| fluent but incomplete | Output is grammatical-looking but misses key content. | Discuss BLEU/chrF and qualitative mismatch. |
| correct/paraphrase | At least one model gives a strong faithful paraphrase. | Provide positive examples. |
