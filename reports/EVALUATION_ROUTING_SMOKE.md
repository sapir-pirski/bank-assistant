# RAG Evaluation Results

Generated at: `2026-06-30T09:45:15.386259+00:00`

## Experiment Design

- Mode: `full`
- Temperature requested: `0.01`
- Judge model: `gpt-4.1`
- Embedding models: `text-embedding-3-large`
- LLM models: `gpt-5.4`
- Similarity metrics: `cosine`
- Top-k values: `3`
- Evaluation questions: `1`
- Raw JSON: `reports/evaluation_routing_smoke.json`

## Best Configurations

| Selection | Embedding | Similarity | Top K | LLM | Avg Score | Pass Rate | Avg Latency | Temp Fallbacks |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Best overall | text-embedding-3-large | cosine | 3 | gpt-5.4 | 5 | 1 | 7654 | 0 |
| Best with temperature honored | text-embedding-3-large | cosine | 3 | gpt-5.4 | 5 | 1 | 7654 | 0 |

## Ranked Results

| Rank | Stage | Embedding | Similarity | Top K | LLM | Grounded | Relevant | Safe | Avg | Pass | Latency | Temp Fallbacks | Errors |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | full | text-embedding-3-large | cosine | 3 | gpt-5.4 | 5 | 5 | 5 | 5 | 1 | 7654 | 0 | 0 |

## Notes

- `Temp Fallbacks` counts answer-generation calls where the model rejected `temperature` and the client retried without it.
- For the final chatbot, prefer the best configuration with zero temperature fallbacks if temperature support is a hard requirement.
- Scores are produced by an LLM judge over retrieved sources, generated answer, and expected behavior.
- Judge responses are requested with a Pydantic-derived JSON schema and validated by `JudgeResult`; the script falls back to plain JSON parsing if structured output is rejected.

## Evaluation Cases

- `atm_fee_israel`: What are the cash withdrawal fees from ATMs in Israel?
