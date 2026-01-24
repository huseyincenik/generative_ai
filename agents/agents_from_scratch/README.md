
# Agents From Scratch

Build a minimal autonomous agent loop with tool-calling using the Moonshot AI Kimi-K2-Thinking model via Hugging Face Inference. This folder contains a notebook that walks through setup, tool schemas, and a custom `Agent` class.

[agents_from_scratch.ipynb](agents_from_scratch.ipynb)

Kaggle version: https://www.kaggle.com/code/huseyincenik/agents-from-scratch

## Used Technologies

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)
![Pydantic](https://img.shields.io/badge/Pydantic-0B5FFF?logo=pydantic&logoColor=white)

## What the Notebook Covers

- Install and authenticate Hugging Face Inference
- Create a tool function (currency exchange)
- Define JSON schema for tool-calling (manual + Pydantic)
- Build a minimal `Agent` loop that executes tool calls
- Run an end-to-end demo

## Quick Start

1. Open the notebook: [agents_from_scratch.ipynb](agents_from_scratch.ipynb)
2. Install dependencies:
   - `huggingface_hub`
   - `pydantic`
3. Set your Hugging Face token when prompted.

## Architecture

```mermaid
flowchart TD
	A[User Question] --> B[Agent Loop]
	B --> C{LLM decides}
	C -->|Tool call| D[Execute Tool Function]
	D --> E[Tool Result]
	E --> B
	C -->|Final answer| F[Assistant Response]
```

```mermaid
sequenceDiagram
	participant U as User
	participant A as Agent
	participant L as LLM (Kimi-K2-Thinking)
	participant T as Tool (Python function)

	U->>A: Ask question
	A->>L: Send messages + tools
	L-->>A: Tool call request
	A->>T: Execute function with args
	T-->>A: Tool output
	A->>L: Append tool output
	L-->>A: Final response
	A-->>U: Answer
```

## Notes

- The LLM is accessed via `InferenceClient` from `huggingface_hub`.
- Tool schemas are provided as JSON (manual) or generated with `pydantic`.
- The agent loop keeps calling the model until it returns a final message.

## Sources

- Hugging Face Hub tutorial: Agents from Scratch
- Intro to Agents (Alejandro AO)

