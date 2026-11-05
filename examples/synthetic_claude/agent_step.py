"""One step of an agent-answered MARICL run (the only command the answering agent uses).

    python agent_step.py --seed S                  # print the pending LLM request (or DONE)
    python agent_step.py --seed S <<'JSON'         # answer the pending request, then print the next
    { ...one JSON object matching the requested schema... }
    JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from claude_backend import install_import_shims  # noqa: E402

install_import_shims()
import maricl_core as mc  # noqa: E402
import run_synthetic as rs  # noqa: E402
from bridge import BridgeLLM, NeedResponse  # noqa: E402

SCHEMAS = {"Hypothesis": mc.Hypothesis, "Correction": mc.Correction, "Critique": mc.Critique}


def show(req: dict, root: str, seed: int) -> None:
    answered = len(os.listdir(os.path.join(root, "responses")))
    props = req["json_schema"]["properties"]
    print(f"=== REQUEST {req['id']} | answer schema: {req['schema']} | {answered} answered so far ===")
    print("[SYSTEM PROMPT]")
    print(req["system"])
    print("\n[USER PROMPT]")
    print(req["user"])
    print("\n[ANSWER FORMAT] one JSON object with exactly these fields:")
    for name, spec in props.items():
        kind = spec.get("type", "string")
        if kind == "array":
            kind = "array of strings"
        print(f"  - {name} ({kind}): {spec.get('description', '')}")
    print(f"\nAnswer with:\npython agent_step.py --seed {seed} <<'JSON'\n{{ ... }}\nJSON")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--T", type=int, default=10)
    args = ap.parse_args()
    root = os.path.join(HERE, "bridge", f"seed{args.seed}")
    os.makedirs(os.path.join(root, "responses"), exist_ok=True)
    pending_path = os.path.join(root, "pending.json")

    answer_text = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read().strip()
    if answer_text:
        if not os.path.exists(pending_path):
            print("No pending request. Run without input to see it.")
            return 1
        pending = json.load(open(pending_path))
        try:
            obj = json.loads(answer_text)
            SCHEMAS[pending["schema"]].model_validate(obj)
        except Exception as exc:  # malformed JSON or schema mismatch: ask again
            print(f"ANSWER REJECTED for request {pending['id']}: {exc}\nFix the JSON and answer again.\n")
            show(pending, root, args.seed)
            return 2
        with open(os.path.join(root, "responses", f"{pending['id']}.json"), "w") as fh:
            json.dump(obj, fh, indent=2)
        os.remove(pending_path)

    counter = {"n": 0}
    llm = BridgeLLM(root, "base", counter)
    llm_creative = BridgeLLM(root, "encoder", counter)
    try:
        rs.run_experiment(args.seed, args.T, llm, llm_creative,
                          out_dir=os.path.join(HERE, "results_agent"), tag=f"seed{args.seed}",
                          fit_log=os.path.join(root, "fit_log.txt"))
    except NeedResponse as need:
        with open(pending_path, "w") as fh:
            json.dump(need.request, fh, indent=2)
        show(need.request, root, args.seed)
        return 0
    print(f"=== DONE: all LLM requests for seed {args.seed} are answered. Stop here. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
