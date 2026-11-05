# Prompt for an agent that answers MARICL's LLM requests

This is the prompt each Claude Code session received for the recorded classification runs, one session per seed. Replace `TASK` with `regression` or `classification`, `SEED` with the seed number, and `PYTHON` with the path to your Python interpreter. Run the session from this folder.

The regression runs used the same setup, as described in Appendix E of the paper. Their exact prompt text was not kept.

The prompts that MARICL itself sends (encoder, decoder, critique, and refinement) are in `maricl_core.py`. Every one actually sent in the recorded runs is saved under `recorded_answers/*/seed*/requests/`.

---

```text
You are acting as the LLM inside a tabular machine learning pipeline called MARICL. The pipeline pauses at each LLM call and gives you one request. You answer it with one JSON object, and the pipeline then gives you the next request. There are about 80 requests.

To see the current request, run exactly this command:

PYTHON agent_step.py --task TASK --seed SEED </dev/null

To answer it, run the same command but replace `</dev/null` with a heredoc holding your JSON, like this:

PYTHON agent_step.py --task TASK --seed SEED <<'JSON'
{ ...your one JSON object... }
JSON

That command records your answer and prints the next request. Each request shows a system prompt, a user prompt, and the JSON fields it wants.

Rules:
1. This command is the only thing you may run. Do not read, list, or search any files, and do not run any other code. Do your reasoning yourself, without calculating in code.
2. Answer each request from its own content, plus what you learned from earlier requests in this session. Do your honest best on each one.
3. Keep going until the output says DONE. If it says ANSWER REJECTED, fix your JSON and answer again.
4. Do not stop early or ask questions. Answer one request per command.

When you see DONE, reply with two lines: how many requests you answered, and whether you ran any command other than the one above.
```

---

**Keep the answering agent blind.** `regression.py` and `classification.py` contain the true functions. For the recorded classification runs, the sessions worked from a copy of this folder in which those two files were compiled to `.pyc` without their source. Do the same, or keep the source out of the agent's reach some other way.

When a session is interrupted, for example by a usage limit, its answers are already saved. Send the same session a message to continue: it resumes at the next unanswered request.
