"""Fast (System 1) LLM bridge: the same file queue as jev_bridge.py, answered by a local Ollama
model with thinking OFF and a short answer budget (one JSON object, no reasoning text).

Host side:  python s1_bridge.py <queue_dir> <ollama model>     e.g. qwen3:8b
            a "+think" suffix (qwen3:4b+think) answers with thinking ON and a larger budget, for the
            thinking on/off comparison; the answer format is the same.
"""

import json
import os
import sys
import urllib.request

import jev_bridge

PROMPT = """You are the storage controller for a CLIO deployment. Read the storage system context and answer every question.

{state}

Questions:
{questions}

Reply with only a JSON object mapping each question id to exactly one listed option key."""


def options_of(q):
    return q.get("options") or q.get("criteria") or q.get("levels")


def fast_answer(body, model, url=None):
    url = url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/chat"
    think = model.endswith("+think")
    model = model.removesuffix("+think")
    qtext = []
    for q in body["questions"]:
        o = options_of(q)
        opts = "; ".join(f'"{k}": {v}' for k, v in o.items())
        qtext.append(f'- id "{q["id"]}" (choose one of {list(o)}): {q["question"]}\n  options: {opts}')
    req = {"model": model, "stream": False, "think": think,
           "messages": [{"role": "user", "content": PROMPT.format(state=body["state"], questions="\n".join(qtext))}],
           "options": {"temperature": 0, "num_ctx": 16384,     # one size: no reload between modes
                       "num_predict": 8192 if think else 200}}
    if not think:
        req["format"] = "json"      # constrained decoding would cut the thinking off
    r = urllib.request.Request(url, data=json.dumps(req).encode(), method="POST",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1800) as f:
        content = json.loads(f.read().decode())["message"]["content"]
    try:
        j = json.loads(content[content.index("{"):content.rindex("}") + 1])
    except ValueError:
        j = {}
    answers = []
    for q in body["questions"]:
        v = j.get(q["id"])
        v = str(v).lower() if isinstance(v, bool) else v
        if v in options_of(q):
            answers.append({"id": q["id"], "value": v, "top_probability": 1.0,
                            "distribution": {k: float(k == v) for k in options_of(q)}})
        else:
            answers.append({"id": q["id"], "value": None, "top_probability": 0.0, "distribution": {}})
    return {"results": [{"answers": answers}]}


if __name__ == "__main__":
    jev_bridge.ollama_answer = lambda body, model: fast_answer(body, model)
    jev_bridge.serve(sys.argv[1], "ollama:" + sys.argv[2])
