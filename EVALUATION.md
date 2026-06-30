# RAG Evaluation Results

Generated at: `2026-06-30T08:58:48.051967+00:00`

## Experiment Design

- Mode: `staged`
- Temperature requested: `0.01`
- Judge model: `gpt-4.1`
- Embedding models: `text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002`
- LLM models: `gpt-5.5, gpt-5.4, gpt-4.1`
- Similarity metrics: `cosine, l2, ip`
- Top-k values: `3, 5, 8`
- Evaluation questions: `5`
- Raw JSON: `reports/evaluation_staged.json`

The run used the staged strategy: first compare retrieval configurations with one LLM, then test the top retrieval configurations across all LLMs. This keeps cost bounded while still testing every requested dimension.

## Best Configurations

| Selection | Embedding | Similarity | Top K | LLM | Avg Score | Pass Rate | Avg Latency | Temp Fallbacks |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Best overall | text-embedding-3-large | cosine | 3 | gpt-5.4 | 4.87 | 1 | 3061 | 0 |
| Best with temperature honored | text-embedding-3-large | cosine | 3 | gpt-5.4 | 4.87 | 1 | 3061 | 0 |

## Ranked Results

| Rank | Stage | Embedding | Similarity | Top K | LLM | Grounded | Relevant | Safe | Avg | Pass | Latency | Temp Fallbacks | Errors |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | stage2_llm | text-embedding-3-large | cosine | 3 | gpt-5.4 | 4.8 | 4.8 | 5 | 4.87 | 1 | 3061 | 0 | 0 |
| 2 | stage2_llm | text-embedding-3-large | cosine | 3 | gpt-5.5 | 4.8 | 4.8 | 5 | 4.87 | 1 | 3386 | 0 | 0 |
| 3 | stage1_retrieval | text-embedding-3-small | cosine | 8 | gpt-4.1 | 4.8 | 4.6 | 5 | 4.8 | 0.8 | 1897 | 0 | 0 |
| 4 | stage1_retrieval | text-embedding-3-large | cosine | 3 | gpt-4.1 | 4.8 | 4.6 | 5 | 4.8 | 0.8 | 2297 | 0 | 0 |
| 5 | stage1_retrieval | text-embedding-3-small | ip | 8 | gpt-4.1 | 4.8 | 4.6 | 5 | 4.8 | 0.8 | 2330 | 0 | 0 |
| 6 | stage2_llm | text-embedding-3-large | cosine | 3 | gpt-4.1 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 1781 | 0 | 0 |
| 7 | stage1_retrieval | text-embedding-3-large | ip | 5 | gpt-4.1 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 1830 | 0 | 0 |
| 8 | stage1_retrieval | text-embedding-3-small | cosine | 5 | gpt-4.1 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 1880 | 0 | 0 |
| 9 | stage1_retrieval | text-embedding-3-large | cosine | 5 | gpt-4.1 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 1891 | 0 | 0 |
| 10 | stage1_retrieval | text-embedding-3-small | cosine | 3 | gpt-4.1 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 1933 | 0 | 0 |
| 11 | stage1_retrieval | text-embedding-ada-002 | ip | 5 | gpt-4.1 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 1980 | 0 | 0 |
| 12 | stage1_retrieval | text-embedding-3-small | ip | 3 | gpt-4.1 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 2071 | 0 | 0 |
| 13 | stage2_llm | text-embedding-3-small | cosine | 8 | gpt-4.1 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 2074 | 0 | 0 |
| 14 | stage1_retrieval | text-embedding-ada-002 | cosine | 3 | gpt-4.1 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 2149 | 0 | 0 |
| 15 | stage1_retrieval | text-embedding-ada-002 | cosine | 5 | gpt-4.1 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 2281 | 0 | 0 |
| 16 | stage2_llm | text-embedding-3-small | ip | 8 | gpt-4.1 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 2320 | 0 | 0 |
| 17 | stage2_llm | text-embedding-3-small | ip | 8 | gpt-5.4 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 2497 | 0 | 0 |
| 18 | stage2_llm | text-embedding-3-small | cosine | 8 | gpt-5.4 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 2835 | 0 | 0 |
| 19 | stage2_llm | text-embedding-3-small | cosine | 8 | gpt-5.5 | 4.6 | 4.6 | 5 | 4.73 | 0.8 | 4018 | 0 | 0 |
| 20 | stage1_retrieval | text-embedding-3-large | ip | 3 | gpt-4.1 | 4.6 | 4.4 | 5 | 4.67 | 0.8 | 1841 | 0 | 0 |
| 21 | stage1_retrieval | text-embedding-3-large | ip | 8 | gpt-4.1 | 4.6 | 4.4 | 5 | 4.67 | 0.8 | 1895 | 0 | 0 |
| 22 | stage1_retrieval | text-embedding-ada-002 | l2 | 5 | gpt-4.1 | 4.6 | 4.4 | 5 | 4.67 | 0.8 | 1966 | 0 | 0 |
| 23 | stage1_retrieval | text-embedding-ada-002 | ip | 3 | gpt-4.1 | 4.6 | 4.4 | 5 | 4.67 | 0.8 | 1966 | 0 | 0 |
| 24 | stage1_retrieval | text-embedding-ada-002 | ip | 8 | gpt-4.1 | 4.6 | 4.4 | 5 | 4.67 | 0.8 | 1978 | 0 | 0 |
| 25 | stage1_retrieval | text-embedding-3-small | ip | 5 | gpt-4.1 | 4.6 | 4.4 | 5 | 4.67 | 0.8 | 2094 | 0 | 0 |
| 26 | stage1_retrieval | text-embedding-3-large | cosine | 8 | gpt-4.1 | 4.6 | 4.4 | 5 | 4.67 | 0.8 | 2179 | 0 | 0 |
| 27 | stage1_retrieval | text-embedding-ada-002 | l2 | 8 | gpt-4.1 | 4.6 | 4.4 | 5 | 4.67 | 0.8 | 2194 | 0 | 0 |
| 28 | stage1_retrieval | text-embedding-ada-002 | l2 | 3 | gpt-4.1 | 4.6 | 4.4 | 5 | 4.67 | 0.8 | 2445 | 0 | 0 |
| 29 | stage1_retrieval | text-embedding-ada-002 | cosine | 8 | gpt-4.1 | 4.6 | 4.4 | 5 | 4.67 | 0.8 | 2804 | 0 | 0 |
| 30 | stage2_llm | text-embedding-3-small | ip | 8 | gpt-5.5 | 4.6 | 4.4 | 5 | 4.67 | 0.8 | 4128 | 0 | 0 |
| 31 | stage1_retrieval | text-embedding-3-small | l2 | 8 | gpt-4.1 | 4.6 | 4.2 | 5 | 4.6 | 0.6 | 1450 | 0 | 0 |
| 32 | stage1_retrieval | text-embedding-3-large | l2 | 5 | gpt-4.1 | 4.4 | 4.2 | 5 | 4.53 | 0.6 | 1381 | 0 | 0 |
| 33 | stage1_retrieval | text-embedding-3-small | l2 | 3 | gpt-4.1 | 4.4 | 4.2 | 5 | 4.53 | 0.6 | 1509 | 0 | 0 |
| 34 | stage1_retrieval | text-embedding-3-large | l2 | 3 | gpt-4.1 | 4.4 | 4.2 | 5 | 4.53 | 0.6 | 2368 | 0 | 0 |
| 35 | stage1_retrieval | text-embedding-3-small | l2 | 5 | gpt-4.1 | 4.4 | 4 | 5 | 4.47 | 0.6 | 1501 | 0 | 0 |
| 36 | stage1_retrieval | text-embedding-3-large | l2 | 8 | gpt-4.1 | 4.4 | 4 | 5 | 4.47 | 0.6 | 1622 | 0 | 0 |

## Notes

- `Temp Fallbacks` counts answer-generation calls where the model rejected `temperature` and the client retried without it.
- For the final chatbot, prefer the best configuration with zero temperature fallbacks if temperature support is a hard requirement.
- Scores are produced by an LLM judge over retrieved sources, generated answer, and expected behavior.

## Evaluation Cases

- `atm_fee_israel`: What are the cash withdrawal fees from ATMs in Israel?
- `cash_deposit`: Can I deposit cash into my ONE ZERO account?
- `one_plus_fees`: What securities trading fees apply to the ONE PLUS plan?
- `short_selling`: Can I short sell securities through ONE ZERO?
- `investment_advice`: Should I buy Apple stock today?
