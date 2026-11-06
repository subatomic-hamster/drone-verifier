#!/usr/bin/env python3
"""Fail (exit 1) if GPL/LGPL/AGPL-licensed code is found in fetched deps.

Usage: check_licenses.py [DIR]   (default: build/_deps)

Scans files named LICENSE*/LICENCE*/COPYING*/COPYRIGHT*/UNLICENSE*/NOTICE* (any extension,
any depth). Decision rule, to avoid false positives without weakening the check:
  1. A file containing the GNU license BODY (heading "GNU [Lesser|Library|Affero] GENERAL
     PUBLIC LICENSE" together with "TERMS AND CONDITIONS") IS a GPL-family license -> FAIL.
  2. A file that merely MENTIONS GPL/LGPL/AGPL (dual-license choice like "MIT or GPL", or an
     exception clause) is OK only if it also carries a permissive grant (MIT/BSD/Apache/zlib/
     ISC/Unlicense/public domain); the mention is printed as a NOTE for human review.
  3. A file that mentions GPL-family and has no permissive grant -> FAIL.
Not covered: GPL code whose license file is missing or named otherwise (e.g. only an SPDX
header in sources); review new deps by hand when adding them to cmake/deps.cmake.
Known case: mavlink/c_library_v2 (generated MAVLink C headers, fetched for dvsim-px4-bridge) ships no license file at all,
so this scan cannot see it. Its terms come from the generator repo's COPYING
(https://github.com/mavlink/mavlink/blob/master/COPYING): the generator is (L)GPL v3, but "as an exception" a product that
embeds the generator OUTPUT may be redistributed under the MIT license. We only fetch and compile the generated headers,
never the generator. Reviewed by hand; see cmake/deps.cmake and docs/research/px4-sitl.md.
"""

import re
import sys
from pathlib import Path

NAME = re.compile(r"(licen[sc]e|copying|copyright|unlicense|notice)", re.IGNORECASE)
# Explicit, path-exact exclusions (never a pattern): test-only trees that are never compiled,
# linked or shipped. ccd's src/testsuites/cu is the LGPL "CU" unit-test harness; ccd only adds
# it when BUILD_TESTING and not CCD_HIDE_ALL_SYMBOLS (MuJoCo hides symbols) and the generated
# build.ninja has no testsuites target (verified at M0). Re-verify when bumping MuJoCo/ccd.
# Eigen 3.4.0 (MPL-2.0, header-only, dog WBC in C++): the archive ships the texts of the licenses of its few LGPL files
# (COPYING.GPL / COPYING.LGPL / bench/btl/COPYING, COPYING.README, the MPL-2.0 text that names the GPL as a "Secondary License",
# scripts/relicense.py). The LGPL code is in unsupported/ and Eigen/src/IterativeLinearSolvers/IncompleteLUT.h, and is never included:
# examples/dog/loco/cpp/CMakeLists.txt compiles with -DEIGEN_MPL2_ONLY, which turns an #include of any LGPL Eigen file into a
# compile error (Eigen's own guarantee, COPYING.README). Path-exact; re-verify when bumping Eigen.
EXCLUDE = (
    "ccd-src/src/testsuites/cu/",
    "eigen-src/COPYING.GPL",
    "eigen-src/COPYING.LGPL",
    "eigen-src/COPYING.MPL2",
    "eigen-src/COPYING.README",
    "eigen-src/bench/btl/COPYING",
    "eigen-src/scripts/relicense.py",
)
# Path-exact, content-checked justification (never a pattern). slangc (build-time tool, M9) ships LICENSE and
# LICENSES/LLVM-exception.txt: Apache-2.0 plus the "LLVM Exceptions to the Apache 2.0 License", whose second paragraph
# says that when the Software is combined with GPLv2 software a conflicting Apache section may be waived. That is a
# permission granted to the user (it adds a compatibility, it does not impose GPL on anything) and no GPL license text
# is in either file. Accepted only if the exact clause title is present and no GNU license body is. The slang archive's
# third-party-notices/ (glslang's vestigial Bison-exception GPL text) is NOT extracted, see cmake/deps.cmake.
LLVM_EXCEPTION_FILES = (
    "slang-bin/LICENSE",
)  # the scanner only visits LICENSE*/COPYING*/... file names
LLVM_EXCEPTION_TITLE = re.compile(r"LLVM Exceptions to the Apache 2\.0 License")
BODY = re.compile(
    r"GNU\s+(LESSER\s+|LIBRARY\s+|AFFERO\s+)?GENERAL\s+PUBLIC\s+LICENSE", re.IGNORECASE
)
TERMS = re.compile(r"TERMS\s+AND\s+CONDITIONS", re.IGNORECASE)
MENTION = re.compile(
    r"\b(A|L)?GPL(v?[23])?\b|GNU\s+(Lesser|Library|Affero)?\s*General\s+Public",
    re.IGNORECASE,
)
PERMISSIVE = [
    ("Apache-2.0", re.compile(r"Apache License\s+Version 2\.0", re.IGNORECASE)),
    ("MIT", re.compile(r"Permission is hereby granted, free of charge", re.IGNORECASE)),
    (
        "zlib",
        re.compile(
            r"warranty\.\s+In no event will the authors be held liable", re.IGNORECASE
        ),
    ),
    (
        "BSD",
        re.compile(r"Redistribution and use in source and binary forms", re.IGNORECASE),
    ),
    (
        "ISC",
        re.compile(
            r"Permission to use, copy, modify, and/or distribute", re.IGNORECASE
        ),
    ),
    ("Unlicense/PD", re.compile(r"public domain|unlicense", re.IGNORECASE)),
]


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "build/_deps")
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and NAME.search(p.name)
        and not any(x in p.as_posix() for x in EXCLUDE)
    )
    if not files:
        print(f"error: no license files under {root} (wrong dir?)", file=sys.stderr)
        return 2
    bad = 0
    for p in files:
        text = p.read_text(errors="replace")
        found = [n for n, rx in PERMISSIVE if rx.search(text)]
        label = ",".join(found) or "unrecognized"
        if BODY.search(text) and TERMS.search(text):
            print(f"FAIL {p}: GPL-family license text")
            bad += 1
        elif p.as_posix().endswith(
            LLVM_EXCEPTION_FILES
        ) and LLVM_EXCEPTION_TITLE.search(text):
            print(
                f"NOTE {p}: [Apache-2.0 WITH LLVM-exception] GPLv2 compatibility clause only; reviewed"
            )
        elif MENTION.search(text):
            if found:
                print(
                    f"NOTE {p}: [{label}] mentions GPL-family (dual/exception); review"
                )
            else:
                print(f"FAIL {p}: mentions GPL-family with no permissive grant")
                bad += 1
        else:
            print(f"ok   {p}: [{label}]")
    print(f"{len(files)} license files scanned, {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
