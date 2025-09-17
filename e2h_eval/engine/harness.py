# engine/harness.py
import multiprocessing as mp, tempfile, os, time, traceback
from contextlib import redirect_stdout, redirect_stderr
import io

from .reliability_guard import reliability_guard

def _escape_triple_single_quotes(s: str) -> str:
    return s.replace("'''", "\\'\\'\\'")

def run_one(problem: dict, completion_src: str, timeout_s: float = 3.0) -> dict:
    """
    problem = {"task_id","prompt","test","entry_point"="check", ...}
    completion_src = candidate script (string)
    Returns: {"status": "passed"|"failed"|"timed out", "ms": int, "err": str|None}
    """
    manager = mp.Manager()
    result = manager.dict()
    prompt = problem.get("prompt","")
    test = problem.get("test","")
    task_id = problem.get("task_id","E2H_TASK")

    def child():
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            reliability_guard()
            program = (
                prompt
                + "\n# ----- BEGIN COMPLETION -----\n"
                + completion_src
                + "\n# ----- END COMPLETION -----\n"
                + test
                + "\n# --- invoke check with candidate source ---\n"
                + "check(r'''"
                + _escape_triple_single_quotes(completion_src)
                + "''')\n"
            )
            f = io.StringIO()
            try:
                with redirect_stdout(f), redirect_stderr(f):
                    exec(program, {})
                result["status"] = "passed"
                result["err"] = None
            except BaseException as e:
                result["status"] = "failed"
                result["err"] = "".join(traceback.format_exception_only(type(e), e))
            finally:
                result["ms"] = int((time.time() - t0) * 1000)

    p = mp.Process(target=child)
    p.start()
    p.join(timeout_s + 0.5)  # wall clock
    if p.is_alive():
        p.kill()
        return {"status": "timed out", "ms": None, "err": None}
    return {"status": result.get("status","failed"), "ms": result.get("ms"), "err": result.get("err")}
