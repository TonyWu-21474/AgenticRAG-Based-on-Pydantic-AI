# Pydantic AI notes

Pydantic AI is a Python agent framework. An agent is declared once with a model
string such as `ollama:qwen3:4b` or `openai:gpt-4o-mini`, and tools are
registered with the `@agent.tool` decorator.

## Tools and dependency injection

A tool whose first parameter is annotated `RunContext[Deps]` receives the
dependency object passed to `agent.run(..., deps=deps)`. Tools without that
parameter can be declared with `@agent.tool_plain`. The docstring of a tool
becomes its description, and an `Args:` section in the docstring becomes the
argument schema the model sees, so parameter descriptions are worth writing.

## Running an agent

```python
result = await agent.run("question", deps=deps)
print(result.output)
```

`result.all_messages()` returns the full conversation, including tool calls and
tool return values, which is how the CLI in this project reports which tools
were used during a turn. Passing `message_history=result.all_messages()` back
into the next run keeps the conversation going.

## Embeddings

The `Embedder` class takes a provider-prefixed model name. It exposes
`embed_documents` for indexing and `embed_query` for retrieval, and both return
an object with an `embeddings` list plus token `usage`. Splitting documents from
queries matters because some embedding models are trained with asymmetric
prefixes.

## Local models

The `ollama:` provider targets a local Ollama server through its
OpenAI-compatible endpoint. The base URL is read from the `OLLAMA_BASE_URL`
environment variable, which must include the `/v1` suffix, for example
`http://127.0.0.1:11434/v1`.
