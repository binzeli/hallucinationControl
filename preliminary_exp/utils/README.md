# Utils

- **`api_caller.py`** — Concurrent API calls with rate-limit retries and optional log-probs.
- **`yaml_parser.py`** — `load_yaml(path)` and `save_yaml(data, path)` for YAML I/O.

## `get_responses_concurrently_with_ids` (api_caller)

Sends POST requests concurrently and returns responses keyed by `request_id`.

| Parameter | Description |
|-----------|-------------|
| `api_url` | API endpoint URL. |
| `payloads` | List of POST bodies. |
| `request_ids` | List of IDs, one per payload (same length). |
| `api_key` | Auth key (run scripts pass from env, e.g. `API_KEY_ALT`). |
| `headers` | Optional request headers. |
| `include_log_prob` | If `True`, payloads should set `logprobs: true`. |
| `return_raw_response` | If `True`, return raw API JSON; run scripts process it. |
| `max_concurrent` | Max concurrent requests (default: 5). |

**Return:** List of `{"request_id": "<id>", "raw_response": ...}` when `return_raw_response=True`, or `{"request_id": "<id>", "error": "..."}` on failure.

**Example:**

```python
import asyncio, os
from utils.api_caller import get_responses_concurrently_with_ids

responses = asyncio.run(get_responses_concurrently_with_ids(
    api_url, payloads, ["id1", "id2"], os.getenv("API_KEY_PROJ"),
    include_log_prob=True, return_raw_response=True
))
```
