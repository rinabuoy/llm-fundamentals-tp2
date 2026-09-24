# LLM Fundamentals: From Basic to AI Agent with Real Practices

Hands-on notebooks for **TP2.0 AI & Data Science, Special Lecture I** (Rina Buoy, PhD). Everything runs on your own laptop.

| # | Notebook | What you'll do |
|---|---|---|
| 01 | [LLM Basics](notebooks/01_llm_basics.ipynb) | Tokenize text, look inside a forward pass, compute the next-token loss by hand, write greedy / top-k / top-p / temperature sampling, and compare a base model with its instruction-tuned twin |
| 02 | [Retrieval-Augmented Generation](notebooks/02_rag.ipynb) | Chunk and embed documents, build a vector-search retriever, score it with Precision@k, MRR and NDCG, then run retrieve → augment → generate |
| 03 | [Tool Calling & MCP](notebooks/03_tool_calling_and_mcp.ipynb) | Call tools via prompting, then natively with JSON Schema, then through local and remote MCP servers |
| 04 | [AI Agents](notebooks/04_ai_agents.ipynb) | Build a ReAct loop and use it for a thermostat agent, a multi-tool agent, an agentic-RAG agent and a web-search agent |

Every notebook is saved **with its outputs**, so you can read it before running anything.

## Setup

1. **Python 3.10+** and the packages:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # or use conda
   pip install -r requirements.txt
   ```
2. **A chat model** for notebooks 02–04. Each of these notebooks starts with a backend switch:
   ```python
   BACKEND = "ollama"  # "ollama" (local) or "api" (OpenAI-compatible cloud API)
   ```
   The rest of the notebook is identical for both options.

   **Option A: Ollama (local, free, private).** Install it from [ollama.com](https://ollama.com), then pull the model:
   ```bash
   ollama pull gemma4:e2b-mlx
   ```
   The `mlx` build only runs on Apple Silicon. On other machines, pull another tool-calling model (for example `ollama pull gemma4:e2b` or `qwen3.5`) and set `OLLAMA_MODEL` to its name.

   **Option B: a cloud API.** Any OpenAI-compatible provider works. Set three environment variables *before* starting Jupyter:
   ```bash
   export LLM_API_KEY="your-key"
   export LLM_API_MODEL="a-tool-calling-model-name"   # from your provider's model list
   export LLM_API_BASE_URL="https://api.openai.com/v1" # OpenAI is the default; see the table below
   ```
   | Provider | `LLM_API_BASE_URL` |
   |---|---|
   | OpenAI | `https://api.openai.com/v1` |
   | Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` |
   | Groq | `https://api.groq.com/openai/v1` |
   | OpenRouter | `https://openrouter.ai/api/v1` |

   Then set `BACKEND = "api"` in the notebook. You can also skip the environment variables and pass the values directly: `llm.configure("api", model="...", base_url="...", api_key="...")`. If no key is set, the notebook asks for one. Never commit your key to the repo.
3. Launch Jupyter **from the `notebooks/` folder**, because the notebooks import `llm_client.py` and `rag_utils.py` and start `simple_mcp.py` from there:
   ```bash
   cd notebooks && jupyter lab
   ```

Small Hugging Face models (SmolLM2-135M, all-MiniLM-L6-v2) download automatically on first use, a few hundred MB in total.

## Repository layout

```
notebooks/
  01_llm_basics.ipynb
  02_rag.ipynb
  03_tool_calling_and_mcp.ipynb
  04_ai_agents.ipynb
  llm_client.py       # llm.chat(): one interface for Ollama or any OpenAI-compatible API
  rag_utils.py        # the retriever from notebook 02, reused in notebook 04
  simple_mcp.py       # tiny FastMCP file-system server used in notebook 03
slides/               # lecture slides (PDF)
reference/TP_full.ipynb   # the original all-in-one practice notebook
```

## Notes

- LLM outputs vary from run to run (and web search results change daily), so your outputs won't match the saved ones exactly.
- `simple_mcp.py` exposes an `edit_file_tool` that can **write files** on your machine. Only run it locally, for learning.
- The second MCP server in notebook 03 is a hosted Hugging Face Space. If it's asleep, the first request can take a minute.
