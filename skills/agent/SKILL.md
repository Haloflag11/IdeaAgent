---
name: agent
description: LLM-powered agent development. Covers prompt engineering, tool/function calling, multi-step reasoning loops, RAG pipelines, and autonomous agent patterns using the OpenAI API.
license: MIT
compatibility: Requires Python 3.11+, openai, tiktoken, numpy, pandas
metadata:
  author: IdeaAgent Team
  version: "1.0"
  category: agent
---

# Agent Skill

Build and evaluate **LLM-powered agents** using the OpenAI API.

## When to Use

Use this skill when:
- Designing prompts, tool schemas, or reasoning chains for LLMs
- Implementing agents that call external tools/functions
- Building retrieval-augmented generation (RAG) pipelines
- Evaluating agent behaviour, instruction-following, or multi-step reasoning

## File Organisation

Always save outputs to organised subdirectories:
```
results/     ← JSON conversation logs, evaluation reports
plots/       ← evaluation charts (.png)
data/        ← datasets, embeddings, retrieval corpora (.json / .csv)
logs/        ← raw API call logs
```

```python
from pathlib import Path
for d in ["results", "plots", "data", "logs"]:
    Path(d).mkdir(exist_ok=True)
```

## Common Patterns

### Basic Chat Completion
```python
import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
model = os.getenv("DEFAULT_MODEL", "gpt-4o")

def chat(messages: list[dict], **kwargs) -> str:
    resp = client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    return resp.choices[0].message.content or ""
```

### Function/Tool Calling
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    }
]

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    tools=tools,
    tool_choice="auto",
)

msg = response.choices[0].message
if msg.tool_calls:
    for tc in msg.tool_calls:
        print(f"Tool called: {tc.function.name}")
        args = json.loads(tc.function.arguments)
        print(f"  args: {args}")
```

### ReAct Agent Loop
```python
def run_react_agent(task: str, tools: dict, max_steps: int = 10) -> str:
    """Simple ReAct (Reason + Act) agent loop."""
    messages = [
        {"role": "system", "content": "Think step by step. Use tools when needed."},
        {"role": "user", "content": task},
    ]
    history = []
    for step in range(max_steps):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tool_schemas, tool_choice="auto"
        )
        msg = response.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            # Final answer
            print(f"Agent finished in {step+1} steps")
            return msg.content or ""
        # Execute each tool call
        for tc in msg.tool_calls:
            fn = tools.get(tc.function.name)
            result = fn(**json.loads(tc.function.arguments)) if fn else "Tool not found"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })
            history.append({"step": step, "tool": tc.function.name, "result": str(result)})
    return "Max steps reached without final answer"
```

### Prompt Evaluation
```python
test_cases = [
    {"prompt": "Translate 'hello' to French", "expected": "bonjour"},
    {"prompt": "What is 2+2?", "expected": "4"},
]

results = []
for tc in test_cases:
    output = chat([{"role": "user", "content": tc["prompt"]}])
    passed = tc["expected"].lower() in output.lower()
    results.append({
        "prompt": tc["prompt"],
        "expected": tc["expected"],
        "output": output,
        "passed": passed,
    })
    print(f"{'PASS' if passed else 'FAIL'}: {tc['prompt']!r}")

accuracy = sum(r["passed"] for r in results) / len(results)
print(f"\nAccuracy: {accuracy:.1%}")
with open("results/eval_report.json", "w") as f:
    json.dump({"accuracy": accuracy, "cases": results}, f, indent=2)
print("Saved results/eval_report.json")
```

### Token Counting
```python
import tiktoken

enc = tiktoken.encoding_for_model(model)

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def estimate_cost(prompt: str, completion: str, rate_per_1k: float = 0.002) -> float:
    total = count_tokens(prompt) + count_tokens(completion)
    return total / 1000 * rate_per_1k
```

## Best Practices
1. Always load credentials from `.env` — never hard-code API keys
2. Log all API calls (prompt + response + latency) to `logs/`
3. Use `temperature=0` for deterministic/evaluation runs
4. Handle `RateLimitError` with exponential back-off
5. Count tokens before sending to avoid unexpected truncation
