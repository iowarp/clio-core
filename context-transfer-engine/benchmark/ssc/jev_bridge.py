"""File-queue bridge between the container and AgentJev on the host (localhost only).

Host side:       python jev_bridge.py serve <queue_dir> [url]
                 python jev_bridge.py serve <queue_dir> ollama:<model>   (System 2 baseline)
Container side:  ask(queue_dir, request) -> response   (request in AgentJev's /api/evaluate shape)

Requests are req_<id>.json; the host worker posts them to AgentJev and writes resp_<id>.json.
No port is opened beyond AgentJev's own loopback listener.
"""

import json
import os
import sys
import time
import uuid
import urllib.request


def submit(queue, request):
    """Queue a request without waiting; returns its id for poll()."""
    rid = uuid.uuid4().hex
    tmp = os.path.join(queue, f".req_{rid}.json")
    with open(tmp, "w") as f:
        json.dump(request, f)
    os.replace(tmp, os.path.join(queue, f"req_{rid}.json"))
    return rid


def poll(queue, rid):
    """The response to request rid, or None while it is still being answered."""
    resp = os.path.join(queue, f"resp_{rid}.json")
    if not os.path.exists(resp):
        return None
    time.sleep(0.02)
    with open(resp) as f:
        out = json.load(f)
    os.remove(resp)
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def ask(queue, request, timeout_s=900):
    rid = submit(queue, request)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = poll(queue, rid)
        if out is not None:
            return out
        time.sleep(0.02)
    raise TimeoutError("no AgentJev response; is `jev_bridge.py serve` running on the host?")


# ---------------------------------------------------------------- System 2 baseline (Ollama)
LLM_PROMPT = """You are the storage controller for a CLIO deployment. Read the storage system context, think it through, then answer every question.

{state}

Questions:
{questions}

After thinking, reply with only a JSON object mapping each question id to its answer: true or false for yes/no questions, exactly one listed option key for choice questions."""


def ollama_answer(body, model, url="http://127.0.0.1:11434/api/chat"):
    """Answer an AgentJev-shaped request with a local LLM; reply in AgentJev's response shape
    (an LLM gives no calibrated probability, so every answer carries probability 1). Thinking is
    capped at 4096 generated tokens; an answer without parsable JSON falls back to "keep"."""
    qtext = []
    for q in body["questions"]:
        if q["type"] == "boolean":
            qtext.append(f'- id "{q["id"]}" (yes/no): {q["question"]}')
        else:
            opts = "; ".join(f'"{k}": {v}' for k, v in q["options"].items())
            qtext.append(f'- id "{q["id"]}" (choose one of {list(q["options"])}): {q["question"]}\n  options: {opts}')
    msg = LLM_PROMPT.format(state=body["state"], questions="\n".join(qtext))
    req = {"model": model, "messages": [{"role": "user", "content": msg}], "stream": False, "think": True,
           "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 4096}}
    r = urllib.request.Request(url, data=json.dumps(req).encode(), method="POST",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f:
        content = json.loads(f.read().decode())["message"]["content"]
    try:
        j = json.loads(content[content.index("{"):content.rindex("}") + 1])
    except ValueError:
        j = {}
    answers = []
    for q in body["questions"]:
        v = j.get(q["id"])
        if q["type"] == "boolean":
            t = v is True or str(v).lower() in ("true", "yes")
            answers.append({"id": q["id"], "distribution": {"true": 1.0 if t else 0.0}})
        elif v in q["options"]:
            answers.append({"id": q["id"], "value": v, "top_probability": 1.0,
                            "distribution": {k: float(k == v) for k in q["options"]}})
        else:   # missing or off-list: no answer, the controller keeps that axis
            answers.append({"id": q["id"], "value": None, "top_probability": 0.0, "distribution": {}})
    return {"results": [{"answers": answers}]}


def serve(queue, url="http://127.0.0.1:8149/api/evaluate"):
    os.makedirs(queue, exist_ok=True)
    print(f"bridge serving {queue} -> {url}", flush=True)
    while True:
        for name in sorted(os.listdir(queue)):
            if not (name.startswith("req_") and name.endswith(".json")):
                continue
            path = os.path.join(queue, name)
            rid = name[4:-5]
            try:
                with open(path) as f:
                    body = json.load(f)
                if url.startswith("ollama:"):
                    t0 = time.time()
                    out = ollama_answer(body, url[7:])
                    print(f"{rid[:8]} answered in {time.time() - t0:.1f} s", flush=True)
                else:
                    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        out = json.loads(r.read().decode())
            except Exception as e:  # report to the caller instead of dying
                out = {"error": str(e)[:500]}
            os.remove(path)
            tmp = os.path.join(queue, f".resp_{rid}.json")
            with open(tmp, "w") as f:
                json.dump(out, f)
            os.replace(tmp, os.path.join(queue, f"resp_{rid}.json"))
        time.sleep(0.02)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "serve":
        serve(sys.argv[2], *sys.argv[3:4])
    else:
        print(__doc__)
