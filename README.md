# LLM Fundamentals: From Basic to AI Agent with Real Practices

Hands-on notebooks for **TP2.0 AI & Data Science, Special Lecture I** (Rina Buoy, PhD). Each notebook follows a block of the [lecture slides](slides/LLM_Fundamentals_From_Basic_to_AI_Agent.pdf) and runs entirely on your own laptop.

| # | Notebook | Slides | What you'll do |
|---|---|---|---|
| 01 | [LLM Basics](notebooks/01_llm_basics.ipynb) | 4–20 | Tokenize text, look inside a forward pass, compute the next-token loss by hand, write greedy / top-k / top-p / temperature sampling, and compare a base model with its instruction-tuned twin |
| 02 | [Retrieval-Augmented Generation](notebooks/02_rag.ipynb) | 21–37 | Chunk documents, build BM25, embedding and hybrid (RRF) retrievers, score them with Precision@k, MRR and NDCG, then run retrieve → augment → generate |
| 03 | [Tool Calling & MCP](notebooks/03_tool_calling_and_mcp.ipynb) | 38–53 | Call tools via prompting, then natively with JSON Schema, then through local and remote MCP servers |
| 04 | [AI Agents](notebooks/04_ai_agents.ipynb) | 54–67 | Build a ReAct loop and use it for a thermostat agent, a multi-tool agent, an agentic-RAG agent and a web-search agent |

Every notebook is saved **with its outputs**, so you can read it before running anything.

## Setup

1. **Python 3.10+** and the packages:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # or use conda
   pip install -r requirements.txt
   ```
2. **Ollama** (for notebooks 02–04): install it from [ollama.com](https://ollama.com), then pull the chat model:
   ```bash
   ollama pull gemma4:e2b-mlx
   ```
   The `mlx` build targets Apple Silicon. On other machines, pull any tool-calling model (for example `ollama pull gemma4:e2b` or `qwen3.5`) and change the `NATIVE_MODEL_ID` / `LLM_MODEL_ID` / `MCP_MODEL_ID` variables at the top of the notebook.
3. Launch Jupyter **from the `notebooks/` folder**, because the notebooks import `rag_utils.py` and start `simple_mcp.py` from there:
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
  rag_utils.py        # the retriever from notebook 02, reused in notebook 04
  simple_mcp.py       # tiny FastMCP file-system server used in notebook 03
slides/               # lecture slides (PDF)
reference/TP_full.ipynb   # the original all-in-one practice notebook
```

## Notes

- LLM outputs vary from run to run (and web search results change daily), so your outputs won't match the saved ones exactly.
- `simple_mcp.py` exposes an `edit_file_tool` that can **write files** on your machine. Only run it locally, for learning.
- The second MCP server in notebook 03 is a hosted Hugging Face Space. If it's asleep, the first request can take a minute.
