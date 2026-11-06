"""Virtual Feetech SCS/STS serial bus: a pty speaking the STS3215 protocol, backed by simulated servo devices.

    port = FeetechPort(sim, servos={1: "/devices/s1", 2: "/devices/s2"})   # sim: dvsim.Sim.load(..., host=True)
    port.device_path        # e.g. /dev/pts/7: open it with unmodified Feetech host code (scservo_sdk, LeRobot, ...)

Protocol (Feetech SCS/STS, 8N1 half-duplex-style, no echo): `FF FF ID LEN INSTR PARAMS.. CHK`, `CHK = ~sum(ID..PARAMS) & 0xFF`,
`LEN = len(PARAMS) + 2`; status `FF FF ID LEN ERR PARAMS.. CHK` (`LEN = len(PARAMS) + 2`). Little-endian words.
Instructions: PING 1, READ 2, WRITE 3, REG_WRITE 4, ACTION 5, RESET 6, SYNC_READ 0x82, SYNC_WRITE 0x83. Broadcast ID 0xFE
never replies. A bad checksum / bad length is dropped (no action, no reply), as a real servo does.
Sources of the register map: LeRobot `motors/feetech/tables.py` (STS_SMS_SERIES_CONTROL_TABLE, sts3215 model number 777,
baud table, sign-magnitude bits: Present_Load 10, Present_Position / Present_Velocity / Goal_Position / Goal_Velocity 15),
the Feetech SDK `scservo_def.SCS_TOHOST(a, b)` (bit b = sign, value = -(a & ~(1 << b))), and parts/arm/feetech-sts3215-c001.toml.

Register map. S = simulated (reads come from / writes go to the sim), P = stored in the port only (RAM / EEPROM image: it
does not influence the physics), R = read-only. A write to an S register becomes a SetAttr on the servo device; at start-up
the port reads the device attrs back into its register image, and a RESET instruction restores the defaults and pushes them.
  0 fw major, 1 fw minor, 3-4 model number 777 .................... P R   (fw version 0.0 is a placeholder)
  5 ID ............................................................ P     (re-keys the port's servo table)
  6 baud index, 7 return delay (x2 us), 8 status level ............ P     (7: wire delay; 8: 0 = no reply to writes; baud
                                                                           reg does not retune the pty: `baud=` does)
  9-12 min / max angle limit ...................................... P     (clamp the goal when max > 0)
  13 max temp (degC) .............................................. S     -> max_temp  (0 = check off)
  14 / 15 max / min input voltage (0.1 V) ......................... S     -> max_voltage / min_voltage  (0 = check off)
  16-17 max torque (0.1 %) ........................................ S     -> max_torque  (duty limit = min with 48-49)
  19 unloading condition (status-bit mask) ........................ S     -> unload_mask (1 voltage, 4 temp, 32 overload)
  21 P, 22 D, 23 I (register units, order as in the table) ........ S     -> p_gain, d_gain, i_gain
  26 / 27 dead zone (counts) ...................................... S     -> deadband = max of the two (the sim's is symmetric)
  35 protection time (10 ms), 36 overload torque (%) .............. S     -> overload_time, overload_load (fraction of FULL duty)
  31-32 homing offset (counts, sign-magnitude bit 11) ............. P     applied by the port: present position = (sim counts -
                                                                           offset) mod cpr and a goal register g is the sim count
                                                                           g + offset (the firmware's "Present_Position =
                                                                           Actual_Position - Homing_Offset", LeRobot feetech.py
                                                                           _get_half_turn_homings; Feetech table addr 31 "Position
                                                                           offset", sign by bit 11 as in LeRobot's encodings table).
                                                                           Sim "actual position" = the encoder count, zero at the
                                                                           zero pose. 0 = factory. Rewrite the goal after changing it.
  18, 20, 24-25, 28-30, 33-34, 37-39 .............................. P     (protection current, holding torque, ...: stored only)
  40 torque enable ................................................ S     -> torque_enable (any nonzero = 1)
  41 acceleration, 44-45 goal time ................................ P
  42-43 goal position (counts, bit 15 = sign) ..................... S     -> goal_position [rad]  (0..cpr-1 = that many counts
                                                                           from the zero pose in the present turn, turn 0 for
                                                                           |counts| < cpr; a negative angle is a bit-15 goal, and
                                                                           needs max angle limit 0 (9-12: limits clamp when max > 0))
  46-47 goal speed (steps/s, bit 15 = sign, magnitude used) ....... S     -> goal_speed [rad/s]  (0 = no profile)
  48-49 torque limit (0.1 %) ...................................... S     -> torque_limit
  55 lock (1 = EEPROM addr < 40 read-only) ........................ P
  56-57 present position (0..cpr-1, wrapped) ...................... S R   (counts - homing offset) mod cpr (the sim encoder is unwrapped)
  58-59 present speed (steps/s, bit 15 sign) ...................... S R   `velocity` (output shaft, rad/s) x cpr / 2 pi
  60-61 present load (0.1 % of full duty, bit 10 sign) ............ S R   `load` (effective duty, fraction) x 1000
  62 present voltage (0.1 V) ...................................... S R   `voltage` attr
  63 present temperature (degC) ................................... S R   `temperature` attr (winding thermal state)
  65 status ....................................................... S R   `alarm` bits: 1 voltage, 4 temperature, 32 overload
  66 moving ....................................................... S R   moving = torque on and |goal - position| > 2 counts
  69-70 present current (6.5 mA) .................................. S R   |`current`| (motor current)
The error byte of every status packet is the servo's `alarm` (bit meanings: the STS status byte of the Feetech memory table,
part file source S4). The alarm latches and releases the torque per register 19; write torque enable 0 (the sim must step
once) to clear it. Rules and limits: src/mujoco/devices/servo.cpp.
Every SetAttr goes through the command bus with source "feetech", so a session with a Host log replays exactly (the log
holds the writes and the Steps; the pty traffic itself and its wall-clock timing are not part of it).
Timing: with `realtime=True` (the Sim running at Host rate 1) each byte of a request and of a reply takes 10 / baud s of
wall clock and the reply additionally waits `return delay x 2 us`; the register is read after the request has arrived.
"""

import os
import select
import termios
import threading
import time
import tty
from math import pi

MODEL_NUMBER = 777  # sts3215 (LeRobot MODEL_NUMBER_TABLE)
BAUDS = {
    0: 1_000_000,
    1: 500_000,
    2: 250_000,
    3: 128_000,
    4: 115_200,
    5: 57_600,
    6: 38_400,
    7: 19_200,
}
PING, READ, WRITE, REG_WRITE, ACTION, RESET, SYNC_READ, SYNC_WRITE = (
    1,
    2,
    3,
    4,
    5,
    6,
    0x82,
    0x83,
)
BROADCAST = 0xFE
_RO = {0, 1, 3, 4, *range(56, 71)}  # fw version, model, present values
_MEM = 128
# Writable registers that drive the sim: (addr, bytes, device attr, register value -> attr, attr -> register value).
# The dead zone is the word 26-27 (CW, CCW): the sim's deadband is symmetric, so the larger one is used.
_REGS = [
    (13, 1, "max_temp", float, round),
    (14, 1, "max_voltage", lambda r: r / 10, lambda v: round(v * 10)),
    (15, 1, "min_voltage", lambda r: r / 10, lambda v: round(v * 10)),
    (16, 2, "max_torque", float, round),
    (19, 1, "unload_mask", float, round),
    (21, 1, "p_gain", float, round),
    (22, 1, "d_gain", float, round),
    (23, 1, "i_gain", float, round),
    (26, 2, "deadband", lambda r: max(r & 255, r >> 8), lambda v: round(v) * 257),
    (35, 1, "overload_time", lambda r: r * 0.01, lambda v: round(v * 100)),
    (36, 1, "overload_load", lambda r: r / 100, lambda v: round(v * 100)),
    (48, 2, "torque_limit", float, round),
]


def checksum(body: bytes) -> int:
    return ~sum(body) & 0xFF


def packet(sid: int, instr_or_err: int, params: bytes = b"") -> bytes:
    body = bytes([sid, len(params) + 2, instr_or_err]) + bytes(params)
    return b"\xff\xff" + body + bytes([checksum(body)])


def _sm(v: int, bit: int) -> int:  # sign-magnitude -> int (SCS_TOHOST)
    return -(v & ~(1 << bit)) if v & (1 << bit) else v


def _to_sm(v: int, bit: int) -> int:  # int -> sign-magnitude (SCS_TOSCS)
    return (-v | (1 << bit)) if v < 0 else v


class _Servo:
    def __init__(self, sid, path, baud_index):
        self.id, self.path = sid, path
        self.mem = bytearray(_MEM)
        self.pending = None  # REG_WRITE buffer: (addr, data)
        self.goal_dirty = False
        self.reset(baud_index)

    def reset(self, baud_index=0):
        m = self.mem
        m[:] = bytes(_MEM)
        m[3:5] = MODEL_NUMBER.to_bytes(2, "little")
        m[5], m[6], m[8] = self.id, baud_index, 1
        m[11:13] = (4095).to_bytes(2, "little")
        m[13], m[14], m[15] = (
            70,
            80,
            40,
        )  # max temp degC (part), max V (assumed 8.0 V), min V 4.0 V (part)
        m[16:18] = m[48:50] = (1000).to_bytes(
            2, "little"
        )  # max torque / torque limit 0.1 % (part)
        m[21], m[22], m[23] = 32, 32, 0  # P, D (part: D estimated), I
        m[26] = m[27] = 1  # dead zones (part)
        m[28:30] = (511).to_bytes(2, "little")
        m[30], m[34], m[35], m[36], m[38] = (
            1,
            20,
            200,
            80,
            200,
        )  # part: protection / overload defaults
        m[19] = (
            36  # unloading condition: temperature + overload (estimated, not published)
        )

    def w(self, a):  # little-endian word
        return self.mem[a] | self.mem[a + 1] << 8

    def put(self, a, v, n=2):
        self.mem[a : a + n] = (v & ((1 << 8 * n) - 1)).to_bytes(n, "little")


class FeetechPort:
    def __init__(
        self,
        sim,
        servos: dict,
        baud: int = 1_000_000,
        cpr: int = 4096,
        symlink=None,
        realtime: bool = True,
    ):
        """sim: a Sim loaded with host=True (thread-safe). servos: {bus id: "/devices/<servo name>"}. cpr: encoder counts
        per revolution (the servo's `encoder_cpr`). symlink: stable path to create for the pty."""
        self.sim, self.baud, self.cpr, self.realtime = sim, baud, cpr, realtime
        bi = {v: k for k, v in BAUDS.items()}.get(baud, 0)
        self.servos = {i: _Servo(i, p, bi) for i, p in servos.items()}
        self.errors = []  # rejected SetAttr replies and dropped frames, for tests / diagnostics
        for s in (
            self.servos.values()
        ):  # the register image starts as the device's actual values
            for addr, n, attr, _, dec in _REGS:
                s.put(addr, dec(self._q(s, attr)), n)
        self._rx = bytearray()
        self.drop = False  # comms fault (dvsim program [[faults]] comms_drop): while True every byte the host sends is discarded
        self._stop = threading.Event()
        self._master, self._slave = os.openpty()
        tty.setraw(self._slave, termios.TCSANOW)
        os.set_blocking(self._master, False)
        self.device_path = os.ttyname(self._slave)
        self.symlink = None
        if symlink is not None:
            self.symlink = os.fspath(symlink)
            if os.path.lexists(self.symlink):
                os.unlink(self.symlink)
            os.symlink(self.device_path, self.symlink)
        self._thread = threading.Thread(
            target=self._serve, name="feetech-port", daemon=True
        )
        self._thread.start()

    # ---- lifecycle -------------------------------------------------------------------------------------------
    def close(self):
        self._stop.set()
        self._thread.join()
        for fd in (self._master, self._slave):
            os.close(fd)
        if self.symlink and os.path.islink(self.symlink):
            os.unlink(self.symlink)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ---- I/O thread ------------------------------------------------------------------------------------------
    def _wire(self, nbytes, extra=0.0):
        if self.realtime:
            time.sleep(nbytes * 10 / self.baud + extra)

    def _serve(self):
        while not self._stop.is_set():
            r, _, _ = select.select([self._master], [], [], 0.05)
            if not r:
                self._rx.clear()  # a stalled partial frame is dropped
                continue
            try:
                data = os.read(self._master, 4096)
            except (BlockingIOError, InterruptedError):
                continue
            if self.drop:
                self._rx.clear()
                continue
            for pkt in self.feed(data):
                self._wire(
                    len(pkt)
                )  # the request's time on the wire; state is sampled after it
                for reply in self.handle(pkt):
                    delay = self._delay(pkt)
                    self._wire(len(reply), delay)
                    try:
                        os.write(self._master, reply)
                    except BlockingIOError:
                        self.errors.append("tx buffer full: reply dropped")

    def _delay(self, pkt):
        s = self.servos.get(pkt[2])
        return (s.mem[7] * 2e-6) if s else 0.0

    # ---- protocol --------------------------------------------------------------------------------------------
    def feed(self, data: bytes) -> list:
        """Append raw bytes, return the complete, checksum-valid request packets (bad frames are dropped)."""
        rx, out = self._rx, []
        rx += data
        while True:
            i = rx.find(b"\xff\xff")
            if i < 0:
                del rx[: max(len(rx) - 1, 0)]  # keep a trailing lone 0xFF
                return out
            del rx[:i]
            if len(rx) < 4:
                return out
            if rx[2] == 0xFF or rx[3] < 2:  # not an ID / length: resync one byte on
                del rx[:1]
                continue
            total = rx[3] + 4
            if len(rx) < total:
                return out
            pkt = bytes(rx[:total])
            if checksum(pkt[2:-1]) != pkt[-1]:
                self.errors.append("bad checksum")
                del rx[:2]
                continue
            del rx[:total]
            out.append(pkt)

    def handle(self, pkt: bytes) -> list:
        """One valid request packet -> list of status packets to send (applies the writes)."""
        sid, instr, params = pkt[2], pkt[4], pkt[5:-1]
        if sid == BROADCAST:
            if instr == SYNC_WRITE and len(params) >= 2:
                addr, n = params[0], params[1]
                body = params[2:]
                for k in range(
                    0, len(body) - n, n + 1
                ):  # ponytail: trailing partial entry ignored
                    s = self.servos.get(body[k])
                    if s:
                        self._write(s, addr, body[k + 1 : k + 1 + n])
            elif instr == SYNC_READ and len(params) >= 2:
                addr, n = params[0], params[1]
                return [
                    r
                    for i in params[2:]
                    if i in self.servos
                    for r in [self._read_reply(self.servos[i], addr, n)]
                    if r
                ]
            elif instr == WRITE and len(params) >= 2:
                for s in list(self.servos.values()):
                    self._write(s, params[0], params[1:])
            elif instr == ACTION:
                for s in list(self.servos.values()):
                    self._action(s)
            return []  # broadcast: no status packet
        s = self.servos.get(sid)
        if s is None:
            return []
        if instr == PING:
            return [packet(sid, self._err(s))]
        if instr == READ and len(params) == 2:
            r = self._read_reply(s, params[0], params[1])
            return [r] if r else []
        if instr == WRITE and len(params) >= 2:
            self._write(s, params[0], params[1:])
        elif instr == REG_WRITE and len(params) >= 2:
            s.pending = (params[0], bytes(params[1:]))
        elif instr == ACTION:
            self._action(s)
        elif instr == RESET:
            s.reset(s.mem[6])
            for addr, n, attr, enc, _ in _REGS:
                self._set(s, attr, enc(s.w(addr) if n == 2 else s.mem[addr]))
        else:
            return []
        return [packet(sid, self._err(s))] if s.mem[8] else []

    def _err(self, s):
        """The error byte of a status packet: the servo's alarm bits."""
        return int(self._q(s, "alarm")) & 0xFF

    def _read_reply(self, s, addr, n):
        if n == 0 or addr + n > _MEM:
            self.errors.append(f"read out of range {addr}+{n}")
            return None
        if addr < 71 and addr + n > 56:
            self._refresh(s)
        return packet(s.id, self._err(s), bytes(s.mem[addr : addr + n]))

    def _action(self, s):
        if s.pending:
            addr, data = s.pending
            s.pending = None
            self._write(s, addr, data)

    # ---- register file <-> simulated device ------------------------------------------------------------------
    def _set(self, s, attr, value):
        r = self.sim.call(
            {
                "source": "feetech",
                "cmd_type": "SetAttr",
                "cmd": {
                    "path": s.path,
                    "attr": attr,
                    "frame": "",
                    "values": [float(value)],
                },
            }
        )
        if not r["ok"]:
            self.errors.append(f"{s.path}.{attr}: {r['message']}")

    def _q(self, s, attr):
        return float(self.sim.query(s.path, attr)[0][0])

    def _write(self, s, addr, data):
        lock = s.mem[55]
        for k, b in enumerate(data):
            a = addr + k
            if a >= _MEM or a in _RO or (lock and a < 40 and a != 55):
                continue
            s.mem[a] = b
        n = len(data)
        hit = lambda a, w=1: addr < a + w and a < addr + n
        for a, w, attr, enc, _ in _REGS:
            if hit(a, w):
                self._set(s, attr, enc(s.w(a) if w == 2 else s.mem[a]))
        if hit(40):
            on = s.mem[40] != 0
            if on and not s.goal_dirty:
                self._set(
                    s, "goal_position", self._q(s, "position")
                )  # enabling holds the present position
            if not on:
                s.goal_dirty = False
            self._set(s, "torque_enable", 1.0 if on else 0.0)
        if hit(
            46, 2
        ):  # speed before position: the profile applies from the first firmware step
            self._set(s, "goal_speed", abs(_sm(s.w(46), 15)) * 2 * pi / self.cpr)
        if hit(42, 2):
            g = _sm(s.w(42), 15)
            lo, hi = s.w(9), s.w(11)
            if hi > 0:
                g = min(max(g, lo), hi)
            off = self._offset(s)
            if off:  # register frame -> encoder frame: raw = goal + offset. A single-turn goal (0..cpr-1) is the
                # representative of (goal + offset) mod cpr nearest the present count (a joint with a limited range never
                # needs the other turn); a signed (bit 15) goal is a plain shift.
                single = 0 <= g < self.cpr
                g += off
                if single:
                    cur = self._q(s, "counts")
                    g += round((cur - g) / self.cpr) * self.cpr
            elif (
                0 <= g < self.cpr
            ):  # single-turn target, reached without wrapping: stay in the current turn. The turn is counts / cpr
                # truncated towards zero (turn 0 for |counts| < cpr, also for negative angles: a floor would put a joint
                # at -65 counts in turn -1 and turn a goal of +100 into -3996, so no joint could cross the zero pose)
                g += int(self._q(s, "counts") / self.cpr) * self.cpr
            s.goal_dirty = True
            self._set(s, "goal_position", g * 2 * pi / self.cpr)
        if hit(5):
            new = s.mem[5]
            if new != s.id and new < BROADCAST:
                self.servos[new] = self.servos.pop(s.id)
                s.id = new

    @staticmethod
    def _offset(
        s,
    ):  # homing offset, regs 31-32: sign-magnitude bit 11 (the 12 bits; higher bits ignored)
        return _sm(s.w(31) & 0xFFF, 11)

    _LIVE = (
        "counts",
        "velocity",
        "load",
        "goal_position",
        "alarm",
        "voltage",
        "temperature",
        "current",
    )

    def _refresh(self, s):
        qm = getattr(
            self.sim, "query_many", None
        )  # a remote sim (dvsim.ctl.RemoteSim): one round trip, not eight
        v = dict(zip(self._LIVE, qm(s.path, self._LIVE))) if qm else None
        q = (lambda a: v[a]) if v else (lambda a: self._q(s, a))
        counts = round(q("counts"))
        speed = round(q("velocity") * self.cpr / (2 * pi))
        load = round(q("load") * 1000)
        goal = q("goal_position") * self.cpr / (2 * pi)
        s.put(56, (counts - self._offset(s)) % self.cpr)
        s.put(58, _to_sm(max(-32767, min(speed, 32767)), 15))
        s.put(60, _to_sm(max(-1000, min(load, 1000)), 10))
        s.put(65, int(q("alarm")), 1)
        s.put(62, round(q("voltage") * 10), 1)
        s.put(63, round(q("temperature")), 1)
        s.put(66, int(s.mem[40] != 0 and abs(goal - counts) > 2), 1)
        s.put(69, round(abs(q("current")) / 0.0065))


# ---- SCS (SCS0009) framing: same packet layout and checksum as STS, but 16-bit registers are BIG-endian (high byte first) and there is no Sync Read ----------
# Sources: rustypot scs0009 docs / Mechaela80/scs0009-servo-driver. Goal block (position 42-43, time 44-45, speed 46-47) as in the Feetech SCS protocol
# manual [est until checked on a real servo, bench B-TILT].
def sts_word(v: int) -> bytes:
    return (v & 0xFFFF).to_bytes(2, "little")


def scs_word(v: int) -> bytes:
    return (v & 0xFFFF).to_bytes(2, "big")


def sync_write(rows: dict, big_endian: bool, addr: int = 42) -> bytes:
    """SYNC_WRITE (0x83) of 16-bit words at addr, {id: (word, ...)}: params = addr, bytes per servo, then id + words. STS little-endian, SCS big-endian."""
    w = scs_word if big_endian else sts_word
    n = len(next(iter(rows.values())))
    params = bytes([addr, 2 * n])
    for sid, words in rows.items():
        params += bytes([sid]) + b"".join(w(x) for x in words)
    return packet(BROADCAST, SYNC_WRITE, params)


def sync_write_goals(
    goals: dict, big_endian: bool, addr: int = 42, time_ms: int = 0, speed: int = 0
) -> bytes:
    """The goal block {id: position counts} as position, time, speed words (6 bytes per servo)."""
    return sync_write(
        {sid: (pos, time_ms, speed) for sid, pos in goals.items()}, big_endian, addr
    )


def read_request(sid: int, addr: int, n: int) -> bytes:
    """READ of n bytes at addr from one servo (the SCS has no SYNC_READ: bus B reads each pod servo by id, one at a time)."""
    return packet(sid, READ, bytes([addr, n]))
