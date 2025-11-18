## Streaming ChatBot Quick Start

`streaming_agent.py` now demonstrates a **pure LLM ChatBot** using `ChatBot.astream()` to emit tokens in real time. 

---

### 1. Environment Setup
1. **Enter the repo and create a virtualenv**
   ```bash
   cd spoon-core
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt

   ```
2. **Configure an OpenRouter (or other OpenAI-compatible) API key**  
   Put the key into `.env`:
   ```
   OPENROUTER_API_KEY=sk-xxxx
   ```
   You can adjust the `llm_provider` and `model_name` in `streaming_agent.py`’s `ChatBot(...)`.

---

### 2. Run the Demo
```bash
python examples/starter/streaming_agent.py
```
The script asks three generic topics in sequence. Each round prints:

- ` Streaming ChatBot response...` – LLM tokens stream in real time.


---

### 3. Customization
| Need | Where to change |
| --- | --- |
| Change model/provider | `ChatBot(...)` inside `create_chatbot()` |
| Customize questions | `PROMPTS` list at the top |
| Tweak output format | Printing logic in `stream_chatbot_response()` |
| Integrate with your app | Reuse `stream_chatbot_response()` or call the `ChatBot.astream()` generator directly |

---

### 4. Embed in Your Project
1. Copy `stream_chatbot_response()` to stream tokens in your asyncio/WebSocket service.
2. Need structured events? See `examples/chatbot_streaming_demo.py` for `astream_events()`/`astream_log()` usage.

---

### 5. Tips
- **`Cleanup failed for ...`** – means a provider isn’t configured; remove it from `config.json` if unused.
- **`websockets.legacy` DeprecationWarning** – upstream warning, safe to ignore.

Enjoy! For tool-enabled or advanced callback demos, check `examples/chatbot_streaming_demo.py`, `examples/neo_toolkit_agent_demo.py`, etc.***
