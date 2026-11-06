"""Event bus of one session: appends to events.jsonl, keeps the list for SSE replay, wakes listeners. Thread-safe; emit() is called from tool threads."""

import json
import threading
import time
from pathlib import Path


class Events:
    def __init__(self, session, path=None):
        self.session = session
        self.items = []
        self.cond = threading.Condition()
        self.f = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "a", buffering=1)

    def emit(self, kind, **fields):
        with self.cond:
            now = round(time.time(), 3)
            # the pinned flight_tick carries its own t (flight seconds), which wins over the unix t; ut is the unix time on every event
            ev = dict(
                seq=len(self.items), t=now, ut=now, session=self.session, kind=kind
            )
            ev.update(fields)
            self.items.append(ev)
            if self.f:
                self.f.write(json.dumps(ev, default=str) + "\n")
            self.cond.notify_all()
        return ev

    def wait_after(self, seq, timeout=10.0):
        """Events with seq >= `seq`, blocking until there is one or the timeout (for the SSE thread)."""
        with self.cond:
            if len(self.items) <= seq:
                self.cond.wait(timeout)
            return self.items[seq:]

    def close(self):
        with self.cond:
            if self.f:
                self.f.close()
                self.f = None
