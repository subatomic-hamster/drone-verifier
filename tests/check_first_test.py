"""first_test on scripted runs through the real Session, on hfbox: `python tests/check_first_test.py`. Partial design_set calls merge, a raising one changes nothing, refused calls do not count,
an appended log counts its last run only, a rerun changes nothing, and no file but results.json is written."""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import first_test as FT
from dv.session import Session

R = ROOT / "reference"
REF = json.loads((R / "design.json").read_text())
GOOD = (R / "controller.py").read_text()
BAD = (R / "do_nothing.py").read_text()
SMALL = "tattu-1300mah-6s-100c-funfly-xt60"  # fails L1: flight time


def session(rd, script):
    """Runs (tool, args) through Session.call, as the server does, then finish."""
    s = Session("T", "check", 0, run_dir=rd)
    for tool, args in script:
        print("  ", tool, json.dumps(s.call(tool, args, getattr(s, tool)))[:110], flush=True)  # fmt: skip
    s.call("finish", dict(note="done"), s.finish)
    return s


def split(design):
    """The same design as two partial updates."""
    keys = list(design)
    return ({k: design[k] for k in keys[:3]}, {k: design[k] for k in keys[3:]})


def hashes(rd):
    return {
        f: hashlib.sha256((Path(rd) / f).read_bytes()).hexdigest()
        for f in ("design.json", "controller.py", "events.jsonl")
    }


def main():
    a, b = split(REF)
    with tempfile.TemporaryDirectory(prefix="dv-ft-") as rd:
        print(
            "reference, first test is fly nominal after partial updates and a malformed one"
        )
        session(rd, [
            ("design_set", dict(choices=a)),
            ("design_set", dict(choices={"esc": "no-such-esc"})),
            ("design_set", dict(choices=b)),
            ("write_controller", dict(code=GOOD)),
            ("fly", dict(scenario="nominal")),
        ])  # fmt: skip
        res = FT.process(rd)
        ft = res["first_test"]
        print(" ", res["first_test_score"], res["score"], ft["trigger"], ft["scenario"], ft["attempt_flown"], ft["cost_usd"], ft["mass_kg"])  # fmt: skip
        assert res["first_test_score"] == 100 and res["score"] == 100
        assert ft["trigger"] == "fly" and ft["scenario"] == "nominal" and ft["attempt_flown"] is True  # fmt: skip
        assert ft["design"] == REF, ft["design"]

        print(
            "appended log: bad first controller, then a good one; refused calls first"
        )
        session(rd, [
            ("design_set", dict(choices=dict(REF, battery=SMALL))),
            ("write_controller", dict(code=BAD)),
            ("fly", dict(scenario="nominal")),  # refused by L1
            ("verify", {}),  # stops at G2
            ("design_set", dict(choices={"battery": REF["battery"]})),
            ("write_controller", dict(code="import os\nclass Controller: pass")),  # rejected: BAD stays
            ("verify", {}),  # the first test
            ("write_controller", dict(code=GOOD)),
            ("verify", {}),
        ])  # fmt: skip
        before = hashes(rd)
        res = FT.process(rd, force=True)
        ft = res["first_test"]
        print(" ", res["first_test_score"], res["score"], ft["trigger"], ft["scenario"], ft["attempt_flown"])  # fmt: skip
        assert res["first_test_score"] == 20 and res["score"] == 100, res["first_test_score"]  # fmt: skip
        assert ft["trigger"] == "verify" and ft["attempt_flown"] is False
        assert ft["design"]["battery"] == REF["battery"]

        print("rerun changes nothing, and writes only results.json")
        again = FT.process(rd)  # already done: no flights
        forced = FT.process(rd, force=True)
        assert again == res and forced["first_test_score"] == 20
        assert hashes(rd) == before
        assert not list(Path(rd).glob("*.tmp"))

    with tempfile.TemporaryDirectory(prefix="dv-ft-") as rd:
        print("never tested, only refused")
        session(rd, [
            ("design_set", dict(choices=dict(REF, battery=SMALL))),
            ("write_controller", dict(code=GOOD)),
            ("fly", dict(scenario="nominal")),
        ])  # fmt: skip
        res = FT.process(rd)
        print(" ", res["first_test_score"], res["score"], res["first_test"]["reason"])
        assert res["first_test_score"] == 0 and res["first_test"]["reason"]
        assert res["score"] == 10  # the final score is untouched: G1 passes, G2 fails

    with tempfile.TemporaryDirectory(prefix="dv-ft-") as rd:
        s = Session("T", "check", 0, run_dir=rd)
        assert FT.process(rd) is None  # live: no final yet
        s.ev.close()
    print("OK")


main()
