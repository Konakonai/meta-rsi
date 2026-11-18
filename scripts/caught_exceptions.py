"""Find runnable answers that caught an exception inside the script and still printed a number.

Reruns every answer that ran on all variants (correct, lucky, silent_wrong) on one non-`low` variant,
under a runner that records EXCEPTION_HANDLED events from solution.py (sys.monitoring, Python 3.12+).

    python scripts/caught_exceptions.py . e4b_think_rules results/zero_shot_v1/caught_rules.json
"""
import json, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
RUNNER = r'''
# Run solution.py and report exceptions that solution.py itself caught (sys.monitoring EXCEPTION_HANDLED).
import json, os, runpy, sys
mon = sys.monitoring
TOOL = 5
mon.use_tool_id(TOOL, "swallow")
caught = []
def on_handled(code, offset, exc):
    if code.co_filename.endswith("solution.py") and len(caught) < 20:
        caught.append(f"{type(exc).__name__}: {str(exc)[:120]}")
mon.register_callback(TOOL, mon.events.EXCEPTION_HANDLED, on_handled)
mon.set_events(TOOL, mon.events.EXCEPTION_HANDLED)
def dump():
    with open(os.environ["SWALLOW_OUT"], "w") as f:
        json.dump(caught, f)
import atexit; atexit.register(dump)
sys.argv = ["solution.py"]
runpy.run_path("solution.py", run_name="__main__")
'''

bench = Path(sys.argv[1]); run = sys.argv[2]; out = sys.argv[3]
sys.path.insert(0, str(bench))
from gsb.harness import extract_code
root = bench / "data/v1"
tasks = {t["task_id"]: t for t in map(json.loads, open(root / "tasks.jsonl"))}
rows = [r for r in map(json.loads, open(bench / f"results/zero_shot_v1/{run}.jsonl"))
        if r["category"] in ("correct", "lucky", "silent_wrong")]
runner = RUNNER
def one(r):
    t = tasks[r["task_id"]]; v = [k for k in t["variants"] if k != "low"][0]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"; shutil.copytree(root / t["variants"][v]["dir"], work)
        (work / "solution.py").write_text(extract_code(r["text"]))
        (work / "_runner.py").write_text(runner)
        env = dict(os.environ, SWALLOW_OUT=str(Path(tmp) / "c.json"))
        try:
            p = subprocess.run([sys.executable, "_runner.py"], cwd=work, capture_output=True, text=True, timeout=60, env=env)
            caught = json.load(open(Path(tmp) / "c.json")) if (Path(tmp) / "c.json").exists() else None
        except subprocess.TimeoutExpired:
            caught, p = None, None
    return {"task_id": r["task_id"], "sample": r["sample"], "family": r["family"], "category": r["category"],
            "variant": v, "caught": caught, "stdout_tail": p.stdout[-300:] if p else None}
with ThreadPoolExecutor(int(os.environ.get("W", "12"))) as ex:
    res = list(ex.map(one, rows))
json.dump(res, open(out, "w"))
print(run, len(res), "with caught exceptions:", sum(bool(x["caught"]) for x in res), "unknown:", sum(x["caught"] is None for x in res))
