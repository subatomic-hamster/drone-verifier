"""Virtual DYNAMIXEL Protocol 2.0 serial bus: a pty speaking the X-series protocol, backed by simulated servo devices.

    port = DynamixelPort(sim, servos={1: "/devices/s1", 2: "/devices/s2"}, models={2: "xl330-m288-t"})
    port.device_path        # e.g. /dev/pts/7: open it with unmodified Dynamixel code (dynamixel_sdk, LeRobot, ...)

Sources (fetched 2026-10-07; the e-Manual says it is moving to docs.robotis.com, same content): protocol
https://emanual.robotis.com/docs/en/dxl/protocol2/ , control tables https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/ and
https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/ (XL330-M077-T: https://emanual.robotis.com/docs/en/dxl/x/xl330-m077/, same
table with other defaults), digest with the CRC table and test vectors: docs/research/dynamixel.md. Models: "xl430-w250-t" (number
1060, default), "xl330-m288-t" (1200), "xl330-m077-t" (1190), chosen per bus id with `models={id: name}`.

Protocol: `FF FF FD 00 | ID | LEN_L LEN_H | INSTR | PARAMS.. | CRC_L CRC_H`, `LEN = len(PARAMS) + 3` counting the stuffed bytes; the
CRC-16 (poly 0x8005, init 0, MSB first, no reflection; e-Manual `update_crc`) covers header .. last parameter (stuffed). Byte
stuffing: every `FF FF FD` after the header in INSTR / PARAMS gets a `FD` appended on TX; on RX the CRC is checked over the received
(stuffed) bytes, then the stuffing is removed. Status `FF FF FD 00 | ID | LEN | 55 | ERR | PARAMS.. | CRC` (ERR bit 7 = Alert while
the Hardware Error Status is not 0; 0x01 Result Fail, 0x02 Instruction Error, 0x03 CRC Error, 0x04 Data Range, 0x05 Data Length,
0x06 Data Limit, 0x07 Access Error). Broadcast ID 0xFE answers only PING, SYNC_READ, BULK_READ (PING in id order; SYNC / BULK READ
in the order of the request); every other instruction is executed silently. Status Return Level (68): 0 = PING only, 1 = PING +
READ (SYNC / BULK READ count as READ), 2 = all. A frame with a bad CRC is not executed; a unicast one to a known id gets a CRC
Error status (the e-Manual lists the error; a real servo's behaviour on a corrupted frame is not documented beyond "waits for the
next header"), a broadcast one nothing. A stalled partial frame is dropped after 50 ms.

Instructions: PING 0x01, READ 0x02, WRITE 0x03, REG_WRITE 0x04 (+ ACTION 0x05, Registered Instruction 69; ACTION without a pending
REG_WRITE = Instruction Error), FACTORY_RESET 0x06 (0xFF all, 0x01 all but ID, 0x02 all but ID and baud; broadcast 0xFF is not
executed, as the e-Manual says; the status goes out first, then the table is reset: RAM and EEPROM, torque off), REBOOT 0x08 (RAM back
to the defaults, EEPROM kept, torque off, Hardware Error Status cleared), CLEAR 0x10 mode 0x01 (Present Position back to one turn,
only while the servo is stopped, else Result Fail; mode 0x02 is Y-series only: Instruction Error), SYNC_READ 0x82, SYNC_WRITE 0x83,
BULK_READ 0x92, BULK_WRITE 0x93. NOT implemented (Instruction Error, silent on broadcast): FAST SYNC / BULK READ 0x8A / 0x9A (one
combined status packet with per-device ERR ID CRC and no stuffing, not needed by the Python SDK's GroupSyncRead / GroupBulkRead),
CONTROL TABLE BACKUP 0x20. A write is validated whole before anything is applied (a real servo may apply a prefix).

Register map (XL430-W250-T; XL330 differences below). S = simulated (reads come from / writes go to the sim), P = stored in the port
only (does not influence the physics; EEPROM / RAM image), R = read-only. A write to an S register becomes a SetAttr on the servo
device (source "dynamixel"). Units are the register units of the e-Manual (pulse = 2 pi / cpr rad, velocity 0.229 rev/min, PWM 0.113 %).
  0 model number (2), 2 model information (4), 6 firmware version .............. P R   (fw 52 is a placeholder; info 0)
  7 ID ........................................................................ P     (re-keys the port's servo table; the status of
                                                                                         the write still carries the old id)
  8 baud rate index ........................................................... P     (does not retune the pty / the wire timing:
                                                                                         `baud=` does; init = index of `baud=`)
  9 return delay time (x2 us) ................................................. P     (wire delay before each status packet)
  10 drive mode, 12 secondary ID, 13 protocol type, 24 moving threshold(*) ..... P     (stored; reverse / time-based profile / torque-on-
                                                                                         by-goal / shadow ID / Protocol 1 are NOT modelled)
  11 operating mode ........................................................... S     (EEPROM; 3 position, 4 extended position, 5
                                                                                         current-based position XL330 only; 0 / 1 / 16
                                                                                         are stored but torque enable is refused)
  20 homing offset (4, pulses) ................................................. P     applied by the port: present = actual + offset
                                                                                         (e-Manual; the opposite sign of feetech.py), goal
                                                                                         g -> sim counts g - offset. Rewrite the goal after
                                                                                         changing it.
  31 temperature limit (degC) .................................................. S     -> max_temp  (0 = check off)
  32 / 34 max / min voltage limit (0.1 V) ...................................... S     -> max_voltage / min_voltage
  36 PWM limit (885 = 100 %) ................................................... S     -> max_torque  (x 1000 / 885)
  38 current limit (XL330, mA) ................................................. P     (no runtime current limit in the device)
  44 velocity limit, 48 / 52 max / min position limit, 60 startup config, 62 PWM slope .. P  (the position limits bound Goal Position
                                                                                         in mode 3: Data Limit Error; velocity limit
                                                                                         bounds Goal Velocity)
  63 shutdown (bit mask) ....................................................... S     -> unload_mask (bits 0, 2, 5 = voltage, temp,
                                                                                         overload, the alarms the device has; 4 and 3 stored)
  64 torque enable ............................................................. S     -> torque_enable (EEPROM locked while 1)
  65 LED, 98 bus watchdog ...................................................... P
  68 status return level, 69 registered instruction (R) ........................ P
  70 hardware error status ..................................................... S R   device `alarm` (1 voltage, 4 overheating, 32 overload:
                                                                                         the same bit values) | latched shutdown bits
  76 / 78 velocity I / P, 88 / 90 feedforward 2nd / 1st ........................ P
  80 / 82 / 84 position D / I / P ............................................... S     -> d_gain / i_gain / p_gain  x D_SCALE / I_SCALE /
                                                                                         P_SCALE (module constants; see below)
  100 goal PWM ................................................................. S     -> torque_limit (|value| x 1000 / 885; <= PWM limit)
  102 goal current (XL330) ..................................................... P     (stored; mode 5 does not limit current)
  104 goal velocity, 108 profile acceleration .................................. P
  112 profile velocity (0.229 rev/min, 0 = infinite) ........................... S     -> goal_speed (velocity-based profile only)
  116 goal position (4) ........................................................ S     -> goal_position [rad]
  120 realtime tick (ms) ....................................................... S R   sim tick (1 tick = 1 ms: fixtures use dt 0.001) mod 32768
  122 moving, 123 moving status ................................................ S R   moving: |velocity| > threshold or profile ongoing;
                                                                                         status bit 0 in position (|goal - count| <= 2),
                                                                                         bit 1 profile ongoing, bits 5:4 rectangular (01)
                                                                                         while ongoing; following error bit 3 stays 0
  124 present PWM (2) .......................................................... S R   `load` (duty) x 885
  126 present load (XL430, 0.1 %) / present current (XL330, mA) ................ S R   `load` x 1000 / `current` x 1000 (signed)
  128 present velocity (4) ..................................................... S R   `velocity`
  132 present position (4) ..................................................... S R   mode 3: (counts + offset) mod cpr; modes 4 / 5:
                                                                                         counts + offset (sim counts: unwrapped, zero at the
                                                                                         zero pose)
  136 velocity trajectory ...................................................... R     0 (stub)
  140 position trajectory ...................................................... S R   `setpoint` as the present position
  144 present input voltage (0.1 V), 146 present temperature (degC) ............ S R   `voltage`, `temperature`
  147 backup ready ............................................................. R     0
XL330: no 38 on XL430; XL330 adds 38 current limit, 62 PWM slope, 102 goal current; 126 is the present current. Defaults per model
as in the e-Manual (XL430: P 640 D 4000 shutdown 52 temp 70 / 72 ...; M288: P 400 D 0 shutdown 53 ...; M077: D 400, velocity limit
1620). Indirect address / data (168.., 224.., 578.., 634..) are not implemented: undefined addresses give an Access Error.

Gain scales: the device runs `u = (p e + i sum e + d de) / pid_scale` (de, sum per firmware step) with an optional `d_scale`. The
firmware law is PWM = (P / 128) e + (I / 65536) sum e + (D / 16) de with 885 = 100 % duty (e-Manual, e in pulses by inference). With
the device calibrated as pid_scale = 128 x 885 and d_scale = 16 x 885 the P and D registers map 1:1 onto p and d and device
i = I x 128 / 65536 = I / 512. P_SCALE, I_SCALE, D_SCALE are those factors; a calibration that changes the device's pid_scale
changes them. At start-up the register image is the device's actual values (a mapped register keeps the table default when the
device already holds the equivalent value); FACTORY_RESET / REBOOT push the table defaults.

Position semantics (e-Manual): mode 3 Goal Position is 0..4095 within the Min / Max Position Limit (Data Limit Error outside),
the servo stays in its present turn (it does not cross the 0 / 4095 wrap; use a homing offset to put the work range mid-turn);
modes 4 / 5 take +-1,048,575 (Data Range Error) and ignore the limits. Present Position is, per the e-Manual, a continuous 4-byte
value while the torque is off and an absolute 0..4095 value after torque-on in mode 3: this port reports the wrapped value in
mode 3 at all times (deviation). Enabling the torque holds the present position unless a goal was written while it was off.

Hardware error / shutdown: the device's alarm bits (a tick with torque off clears them, conditions that still hold set them
again) are reported in register 70 and as the Alert bit. A bit in Shutdown (63) latches in the port, clears the torque
(register and device) and refuses Torque Enable = 1 with Result Fail until REBOOT (e-Manual: "REBOOT is the only method").

Modelled vs stubbed. Modelled: modes 3 / 4 position (and 5 as extended position), torque, goal, profile velocity, position PID,
PWM limit / goal PWM as the duty limit, voltage / temperature limits and shutdown, present position / velocity / PWM / load /
current / voltage / temperature / moving, homing offset, return delay, status return level, error codes, all instructions above.
Stubbed (stored with the right default, size, access and range; no physics): Operating Modes 0 / 1 / 16 (the device has a position
firmware only: the register accepts them, Torque Enable = 1 is then refused with Result Fail), Goal Current / Current Limit (the
device's i_limit is config-only: mode 5 is position mode with the duty limit), Goal Velocity, velocity PI, feedforward gains,
Profile Acceleration (profile = velocity-limited ramp, no acceleration limit), Drive Mode, Bus Watchdog, Secondary ID, Protocol Type,
Startup Configuration, PWM Slope, velocity trajectory, moving status following-error bit, LED, Moving Threshold only for register 122.
Every SetAttr goes through the command bus with source "dynamixel", so a session with a Host log replays exactly (the log holds the
writes and the Steps; the pty traffic itself and its wall-clock timing are not part of it).
Timing: with `realtime=True` (the Sim running at Host rate 1) each byte of a request and of a status takes 10 / baud s of wall
clock and each status additionally waits the servo's `return delay x 2 us`; the register is read after the request has arrived.
"""

import os
import select
import termios
import threading
import time
import tty
from math import pi

BROADCAST = 0xFE
PING, READ, WRITE, REG_WRITE, ACTION, FACTORY_RESET, REBOOT, CLEAR = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    0x10,
)
SYNC_READ, SYNC_WRITE, BULK_READ, BULK_WRITE, STATUS = 0x82, 0x83, 0x92, 0x93, 0x55
OK, FAIL, INSTR_ERR, CRC_ERR, RANGE, LENGTH, LIMIT, ACCESS = range(8)
ALERT = 0x80
BAUDS = {
    0: 9600,
    1: 57_600,
    2: 115_200,
    3: 1_000_000,
    4: 2_000_000,
    5: 3_000_000,
    6: 4_000_000,
    7: 4_500_000,
}
PWM_FULL = 885  # PWM Limit / Goal PWM counts at 100 % duty
VEL_UNIT = 0.229 * 2 * pi / 60  # rad/s per velocity count
P_SCALE, I_SCALE, D_SCALE = (
    1.0,
    128 / 65536,
    1.0,
)  # register -> device p / i / d (device pid_scale 128 x 885, d_scale 16 x 885)
_ALARM = 0x25  # device alarm bits: voltage 1, temperature 4, overload 32 (= Hardware Error Status bits 0, 2, 5)
_MEM = 148
_HDR = b"\xff\xff\xfd\x00"
_MAX_LEN = 4096
FW = 52  # placeholder firmware version
_SIGNED = {
    "homing_offset",
    "goal_pwm",
    "goal_current",
    "goal_velocity",
    "goal_position",
    "present_pwm",
    "present_load",
    "present_current",
    "present_velocity",
    "present_position",
    "velocity_trajectory",
    "position_trajectory",
    "bus_watchdog",
}

# (addr, size, name, access, default, lo, hi); model-dependent entries are filled in by _table().
_MODELS = {
    "xl430-w250-t": dict(
        number=1060,
        bauds=7,
        proto=(1, 2),
        temp=72,
        vmax=140,
        vmin=60,
        vlo=60,
        vhi=140,
        vel=265,
        velhi=1023,
        shutdown=52,
        vi=1000,
        vp=100,
        pd=4000,
        pp=640,
        modes=(1, 3, 4, 16),
        xl330=False,
    ),
    "xl330-m288-t": dict(
        number=1200,
        bauds=6,
        proto=(2, 22),
        temp=70,
        vmax=70,
        vmin=35,
        vlo=31,
        vhi=70,
        vel=445,
        velhi=2047,
        shutdown=53,
        vi=1600,
        vp=180,
        pd=0,
        pp=400,
        modes=(0, 1, 3, 4, 5, 16),
        xl330=True,
    ),
    "xl330-m077-t": dict(
        number=1190,
        bauds=6,
        proto=(2, 22),
        temp=70,
        vmax=70,
        vmin=35,
        vlo=31,
        vhi=70,
        vel=1620,
        velhi=2047,
        shutdown=53,
        vi=400,
        vp=40,
        pd=400,
        pp=400,
        modes=(0, 1, 3, 4, 5, 16),
        xl330=True,
    ),
}


def _table(m):
    t = [
        (0, 2, "model_number", "R", m["number"], 0, 0xFFFF),
        (2, 4, "model_information", "R", 0, 0, 0xFFFFFFFF),
        (6, 1, "firmware_version", "R", FW, 0, 255),
        (7, 1, "id", "RW", 1, 0, 252),
        (8, 1, "baud_rate", "RW", 1, 0, m["bauds"]),
        (9, 1, "return_delay_time", "RW", 250, 0, 254),
        (10, 1, "drive_mode", "RW", 0, 0, 13),
        (11, 1, "operating_mode", "RW", 3, 0, 16),
        (12, 1, "secondary_id", "RW", 255, 0, 255),
        (13, 1, "protocol_type", "RW", 2, *m["proto"]),
        (20, 4, "homing_offset", "RW", 0, -1044479, 1044479),
        (24, 4, "moving_threshold", "RW", 10, 0, 1023),
        (31, 1, "temperature_limit", "RW", m["temp"], 0, 100),
        (32, 2, "max_voltage_limit", "RW", m["vmax"], m["vlo"], m["vhi"]),
        (34, 2, "min_voltage_limit", "RW", m["vmin"], m["vlo"], m["vhi"]),
        (36, 2, "pwm_limit", "RW", PWM_FULL, 0, PWM_FULL),
        (44, 4, "velocity_limit", "RW", m["vel"], 0, m["velhi"]),
        (48, 4, "max_position_limit", "RW", 4095, 0, 4095),
        (52, 4, "min_position_limit", "RW", 0, 0, 4095),
        (60, 1, "startup_configuration", "RW", 0, 0, 3),
        (63, 1, "shutdown", "RW", m["shutdown"], 0, 255),
        (64, 1, "torque_enable", "RW", 0, 0, 1),
        (65, 1, "led", "RW", 0, 0, 1),
        (68, 1, "status_return_level", "RW", 2, 0, 2),
        (69, 1, "registered_instruction", "R", 0, 0, 1),
        (70, 1, "hardware_error_status", "R", 0, 0, 255),
        (76, 2, "velocity_i_gain", "RW", m["vi"], 0, 16383),
        (78, 2, "velocity_p_gain", "RW", m["vp"], 0, 16383),
        (80, 2, "position_d_gain", "RW", m["pd"], 0, 16383),
        (82, 2, "position_i_gain", "RW", 0, 0, 16383),
        (84, 2, "position_p_gain", "RW", m["pp"], 0, 16383),
        (88, 2, "feedforward_2nd_gain", "RW", 0, 0, 16383),
        (90, 2, "feedforward_1st_gain", "RW", 0, 0, 16383),
        (98, 1, "bus_watchdog", "RW", 0, -1, 127),
        (100, 2, "goal_pwm", "RW", PWM_FULL, -PWM_FULL, PWM_FULL),
        (104, 4, "goal_velocity", "RW", 0, -m["velhi"], m["velhi"]),
        (108, 4, "profile_acceleration", "RW", 0, 0, 32767),
        (112, 4, "profile_velocity", "RW", 0, 0, 32767),
        (116, 4, "goal_position", "RW", 0, -1048575, 1048575),
        (120, 2, "realtime_tick", "R", 0, 0, 32767),
        (122, 1, "moving", "R", 0, 0, 1),
        (123, 1, "moving_status", "R", 0, 0, 255),
        (124, 2, "present_pwm", "R", 0, -32768, 32767),
        (
            126,
            2,
            "present_current" if m["xl330"] else "present_load",
            "R",
            0,
            -32768,
            32767,
        ),
        (128, 4, "present_velocity", "R", 0, -(1 << 31), (1 << 31) - 1),
        (132, 4, "present_position", "R", 0, -(1 << 31), (1 << 31) - 1),
        (136, 4, "velocity_trajectory", "R", 0, -(1 << 31), (1 << 31) - 1),
        (140, 4, "position_trajectory", "R", 0, -(1 << 31), (1 << 31) - 1),
        (144, 2, "present_input_voltage", "R", 0, 0, 0xFFFF),
        (146, 1, "present_temperature", "R", 0, 0, 255),
        (147, 1, "backup_ready", "R", 0, 0, 1),
    ]
    if m["xl330"]:
        t += [
            (38, 2, "current_limit", "RW", 1750, 0, 1750),
            (62, 1, "pwm_slope", "RW", 140, 1, 255),
            (102, 2, "goal_current", "RW", 1750, -1750, 1750),
        ]
    return t


# Registers that drive the sim: (register name, device attr, register -> attr, attr -> register).
_MAP = [
    ("temperature_limit", "max_temp", float, round),
    ("max_voltage_limit", "max_voltage", lambda r: r / 10, lambda v: round(v * 10)),
    ("min_voltage_limit", "min_voltage", lambda r: r / 10, lambda v: round(v * 10)),
    (
        "pwm_limit",
        "max_torque",
        lambda r: r * 1000 / PWM_FULL,
        lambda v: round(v * PWM_FULL / 1000),
    ),
    (
        "goal_pwm",
        "torque_limit",
        lambda r: abs(r) * 1000 / PWM_FULL,
        lambda v: round(v * PWM_FULL / 1000),
    ),
    ("shutdown", "unload_mask", lambda r: r & _ALARM, round),
    ("position_p_gain", "p_gain", lambda r: r * P_SCALE, lambda v: round(v / P_SCALE)),
    ("position_i_gain", "i_gain", lambda r: r * I_SCALE, lambda v: round(v / I_SCALE)),
    ("position_d_gain", "d_gain", lambda r: r * D_SCALE, lambda v: round(v / D_SCALE)),
    (
        "profile_velocity",
        "goal_speed",
        lambda r: r * VEL_UNIT,
        lambda v: round(v / VEL_UNIT),
    ),
]


def _mk_crc():
    t = []
    for i in range(256):
        c = i << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x8005) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
        t.append(c)
    return t


_CRC = _mk_crc()


def crc16(data: bytes, crc: int = 0) -> int:
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC[((crc >> 8) ^ b) & 0xFF]
    return crc


def stuff(b: bytes) -> bytes:
    return bytes(b).replace(b"\xff\xff\xfd", b"\xff\xff\xfd\xfd")


def unstuff(b: bytes) -> bytes:
    return bytes(b).replace(b"\xff\xff\xfd\xfd", b"\xff\xff\xfd")


def packet(sid: int, instr: int, params: bytes = b"") -> bytes:
    """An instruction (or, with instr = STATUS and the error byte first in params, a status) packet: stuffed, LEN, CRC."""
    body = stuff(bytes([instr]) + bytes(params))
    p = _HDR + bytes([sid]) + (len(body) + 2).to_bytes(2, "little") + body
    return p + crc16(p).to_bytes(2, "little")


def status(sid: int, err: int, params: bytes = b"") -> bytes:
    return packet(sid, STATUS, bytes([err]) + bytes(params))


def _le(b, signed=False):
    return int.from_bytes(bytes(b), "little", signed=signed)


class _Servo:
    def __init__(self, sid, path, model, baud_index):
        self.id, self.path, self.model, self.m = sid, path, model, _MODELS[model]
        self.regs = {r[2]: r for r in _table(self.m)}
        self.owner = {
            a: r for r in self.regs.values() for a in range(r[0], r[0] + r[1])
        }
        self.mem = bytearray(_MEM)
        self.pending = None  # REG_WRITE buffer: (addr, data)
        self.goal_dirty, self.hw, self.cur, self.shift = False, 0, 0, 0
        self.reset(baud_index=baud_index)

    def v(self, name):
        r = self.regs[name]
        return _le(self.mem[r[0] : r[0] + r[1]], name in _SIGNED)

    def set(self, name, value):
        r = self.regs[name]
        self.mem[r[0] : r[0] + r[1]] = (int(value) & ((1 << 8 * r[1]) - 1)).to_bytes(
            r[1], "little"
        )

    def reset(self, keep_id=False, keep_baud=False, ram_only=False, baud_index=None):
        for r in self.regs.values():
            if (
                (ram_only and r[0] < 64)
                or (keep_id and r[2] == "id")
                or (keep_baud and r[2] == "baud_rate")
            ):
                continue
            self.set(r[2], r[4])
        if baud_index is not None:
            self.set("baud_rate", baud_index)
        if not keep_id and not ram_only:
            self.set("id", self.id if baud_index is not None else 1)
        self.set("goal_pwm", self.v("pwm_limit"))
        if "goal_current" in self.regs:
            self.set("goal_current", self.v("current_limit"))
        self.pending, self.goal_dirty, self.hw, self.shift = None, False, 0, 0


class DynamixelPort:
    def __init__(
        self,
        sim,
        servos: dict,
        baud: int = 1_000_000,
        cpr: int = 4096,
        symlink=None,
        realtime: bool = True,
        models: dict | None = None,
    ):
        """sim: a Sim loaded with host=True (thread-safe). servos: {bus id: "/devices/<servo name>"}. cpr: encoder counts per
        revolution (the servo's `encoder_cpr`). models: {bus id: "xl430-w250-t" | "xl330-m288-t" | "xl330-m077-t"}, default
        xl430-w250-t. symlink: stable path to create for the pty."""
        self.sim, self.baud, self.cpr, self.realtime = sim, baud, cpr, realtime
        models = {i: "xl430-w250-t" for i in servos} | {
            i: str(n).lower() for i, n in (models or {}).items()
        }
        bad = {n for n in models.values() if n not in _MODELS}
        if bad:
            raise ValueError(
                f"unknown Dynamixel model {sorted(bad)}; known: {sorted(_MODELS)}"
            )
        bi = {v: k for k, v in BAUDS.items()}.get(baud, 1)
        self.servos = {i: _Servo(i, p, models[i], bi) for i, p in servos.items()}
        self.errors = []  # rejected SetAttr replies and dropped frames, for tests / diagnostics
        for s in (
            self.servos.values()
        ):  # the register image starts as the device's actual values
            for name, attr, enc, dec in _MAP:
                cur = self._q(s, attr)
                if abs(enc(s.v(name)) - cur) > 1e-9 * max(1.0, abs(cur)):
                    s.set(name, dec(cur))
            s.set("goal_position", self._pos(s, self._q(s, "counts")))
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
            target=self._serve, name="dynamixel-port", daemon=True
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
            for n, sid, instr, params, ok in self.feed(data):
                self._wire(
                    n
                )  # the request's time on the wire; state is sampled after it
                for s, reply in self.handle(sid, instr, params, ok):
                    self._wire(len(reply), s.v("return_delay_time") * 2e-6)
                    try:
                        os.write(self._master, reply)
                    except BlockingIOError:
                        self.errors.append("tx buffer full: reply dropped")

    # ---- protocol --------------------------------------------------------------------------------------------
    def feed(self, data: bytes) -> list:
        """Append raw bytes, return the complete request frames (wire length, id, instruction, unstuffed params, crc ok)."""
        rx, out = self._rx, []
        rx += data
        while True:
            i = rx.find(_HDR)
            if i < 0:
                del rx[: max(len(rx) - 3, 0)]  # keep a possible partial header
                return out
            del rx[:i]
            if len(rx) < 7:
                return out
            ln = rx[5] | rx[6] << 8
            if (
                rx[4] > 252 and rx[4] != BROADCAST
            ) or not 3 <= ln <= _MAX_LEN:  # not an ID / length: resync
                del rx[:4]
                continue
            if len(rx) < 7 + ln:
                return out
            pkt = bytes(rx[: 7 + ln])
            if crc16(pkt[:-2]) != _le(pkt[-2:]):
                self.errors.append("bad crc")
                del rx[:4]
                out.append((len(pkt), pkt[4], 0, b"", False))
                continue
            del rx[: 7 + ln]
            body = unstuff(pkt[7:-2])
            out.append((len(pkt), pkt[4], body[0], body[1:], True))

    def handle(self, sid, instr, params, ok=True) -> list:
        """One request frame -> [(servo, status packet)] to send, in order (applies the writes)."""
        if sid != BROADCAST:
            s = self.servos.get(sid)
            if s is None:
                return []
            if not ok:
                return (
                    [(s, status(sid, CRC_ERR | (ALERT if self._hw(s) else 0)))]
                    if s.mem[68] >= 2
                    else []
                )
            r = self._do(s, instr, params)
            return [(s, r)] if r else []
        if not ok:
            return []
        every = sorted(self.servos.values(), key=lambda s: s.id)
        if instr == PING:
            return [(s, self._do(s, PING, b"")) for s in every]
        if instr in (SYNC_READ, BULK_READ):
            if instr == SYNC_READ and len(params) >= 4:
                reqs = [(i, params[:4]) for i in params[4:]]
            elif instr == BULK_READ:
                reqs = [
                    (params[k], params[k + 1 : k + 5])
                    for k in range(0, len(params) - 4, 5)
                ]
            else:
                return []
            out = [
                (self.servos.get(i), self._do(self.servos[i], READ, p))
                for i, p in reqs
                if i in self.servos
            ]
            return [(s, r) for s, r in out if r]
        if instr == SYNC_WRITE and len(params) >= 4:
            n = _le(params[2:4])
            for k in range(4, len(params) - n, n + 1):
                if params[k] in self.servos:
                    self._do(
                        self.servos[params[k]],
                        WRITE,
                        params[:2] + params[k + 1 : k + 1 + n],
                        False,
                    )
        elif instr == BULK_WRITE:
            k = 0
            while k + 5 <= len(params):
                n = _le(params[k + 3 : k + 5])
                if params[k] in self.servos:
                    self._do(
                        self.servos[params[k]],
                        WRITE,
                        params[k + 1 : k + 3] + params[k + 5 : k + 5 + n],
                        False,
                    )
                k += 5 + n
        elif instr in (WRITE, REG_WRITE, ACTION, REBOOT, CLEAR, FACTORY_RESET):
            if not (
                instr == FACTORY_RESET and params[:1] == b"\xff"
            ):  # broadcast Reset All is not executed
                for s in every:
                    self._do(s, instr, params, False)
        return []  # broadcast: no status packet

    def _hw(self, s):
        """Poll the device alarm: latch shutdown bits (torque cleared), return the Hardware Error Status."""
        a = int(self._q(s, "alarm")) & _ALARM
        s.hw |= a & s.v("shutdown")
        if s.hw and s.mem[64]:
            s.set("torque_enable", 0)
            s.goal_dirty = False
            self._set(s, "torque_enable", 0.0)
        s.cur = s.hw | a
        return s.cur

    def _do(self, s, instr, p, reply=True):
        """Execute one instruction on one servo -> its status packet (None: no status for this instruction / level)."""
        hw, err, data, act = self._hw(s), OK, b"", None
        need = {PING: 0, READ: 1}.get(instr, 2)
        if instr == PING:
            data = bytes(s.mem[0:2] + s.mem[6:7])
        elif instr == READ:
            if len(p) != 4:
                err = INSTR_ERR
            else:
                a, n = _le(p[:2]), _le(p[2:])
                err = (
                    LENGTH
                    if n == 0
                    else ACCESS
                    if any(x not in s.owner for x in range(a, a + n))
                    else OK
                )
                if not err:
                    if a <= 147 and (a + n > 120 or a <= 64 < a + n or a <= 70 < a + n):
                        self._refresh(s)
                    data = bytes(s.mem[a : a + n])
        elif instr in (WRITE, REG_WRITE):
            err = INSTR_ERR if len(p) < 2 else self._check(s, _le(p[:2]), bytes(p[2:]))
            if not err:
                a, d = _le(p[:2]), bytes(p[2:])
                act = (
                    (lambda: self._apply(s, a, d))
                    if instr == WRITE
                    else (lambda: self._register(s, a, d))
                )
        elif instr == ACTION:
            if s.pending is None:
                err = INSTR_ERR
            else:
                a, d = s.pending
                err = self._check(s, a, d)
                act = lambda: (
                    setattr(s, "pending", None),
                    s.set("registered_instruction", 0),
                    None if err else self._apply(s, a, d),
                )
        elif instr == FACTORY_RESET:
            err = RANGE if len(p) != 1 or p[0] not in (0xFF, 1, 2) else OK
            act = None if err else (lambda: self._factory(s, p[0]))
        elif instr == REBOOT:
            act = lambda: self._reboot(s)
        elif instr == CLEAR:
            if len(p) == 5 and p[0] == 1 and bytes(p[1:]) == b"\x44\x58\x4c\x22":
                self._refresh(s)
                err = FAIL if s.mem[122] else OK  # only while stopped
                act = None if err else (lambda: self._clear(s))
            else:
                err = (
                    INSTR_ERR if len(p) == 5 and p[0] == 2 else RANGE
                )  # mode 2 (clear error): Y series only
        else:
            err = INSTR_ERR
        out = (
            status(s.id, err | (ALERT if hw else 0), data)
            if reply and s.mem[68] >= need
            else None
        )
        if act:
            act()
        return out

    # ---- register file ---------------------------------------------------------------------------------------
    def _check(self, s, addr, data):
        """Validate a write of `data` at `addr`; error code or 0. Nothing is applied here."""
        end, regs = addr + len(data), []
        if not data:
            return LENGTH
        for a in range(addr, end):
            r = s.owner.get(a)
            if r is None or r[3] == "R" or (r[0] < 64 and s.mem[64]):
                return ACCESS  # undefined / read-only / EEPROM while the torque is on
            if not regs or regs[-1] is not r:
                regs.append(r)
        if any(addr > r[0] or end < r[0] + r[1] for r in regs):
            return LENGTH
        new = {
            r[2]: _le(data[r[0] - addr : r[0] - addr + r[1]], r[2] in _SIGNED)
            for r in regs
        }
        get = lambda n: new[n] if n in new else s.v(n)
        for name, v in new.items():
            r = s.regs[name]
            if not r[5] <= v <= r[6] or (
                name == "operating_mode" and v not in s.m["modes"]
            ):
                return RANGE
            if (
                name == "goal_position"
                and get("operating_mode") == 3
                and not get("min_position_limit") <= v <= get("max_position_limit")
            ):
                return LIMIT
            if (name == "goal_pwm" and abs(v) > get("pwm_limit")) or (
                name == "goal_velocity" and abs(v) > get("velocity_limit")
            ):
                return LIMIT
            if name == "goal_current" and abs(v) > get("current_limit"):
                return LIMIT
            if (
                name == "torque_enable"
                and v == 1
                and (s.hw & s.v("shutdown") or get("operating_mode") not in (3, 4, 5))
            ):
                return FAIL  # shut down (REBOOT only) or a mode the device has no firmware for
            if name == "id" and v != s.id and v in self.servos:
                return FAIL
        return OK

    def _register(self, s, addr, data):
        s.pending = (addr, data)
        s.set("registered_instruction", 1)

    def _apply(self, s, addr, data):
        s.mem[addr : addr + len(data)] = data
        names = {s.owner[a][2] for a in range(addr, addr + len(data))}
        if (
            "operating_mode" in names
        ):  # e-Manual: a mode change resets the profile and the goal PWM / current
            s.set("profile_velocity", 0)
            s.set("profile_acceleration", 0)
            s.set("goal_pwm", s.v("pwm_limit"))
            if "goal_current" in s.regs:
                s.set("goal_current", s.v("current_limit"))
            names |= {"profile_velocity", "goal_pwm"}
        self._push(s, names)
        if "torque_enable" in names:
            on = s.mem[64] != 0
            if on and not s.goal_dirty:  # enabling holds the present position
                self._set(s, "goal_position", self._q(s, "position"))
                s.set("goal_position", self._pos(s, self._q(s, "counts")))
            if not on:
                s.goal_dirty = False
            self._set(s, "torque_enable", 1.0 if on else 0.0)
        if "goal_position" in names:
            s.goal_dirty = True
            self._set(
                s,
                "goal_position",
                self._goal_counts(s, s.v("goal_position")) * 2 * pi / self.cpr,
            )
        if "id" in names and s.v("id") != s.id:
            self.servos[s.v("id")] = self.servos.pop(s.id)
            s.id = s.v("id")

    def _push(self, s, names):
        for (
            name,
            attr,
            enc,
            _,
        ) in (
            _MAP
        ):  # profile velocity before the goal: the profile applies from the first step
            if name in names:
                self._set(s, attr, enc(s.v(name)))

    def _factory(self, s, opt):
        old_id = s.id
        s.reset(keep_id=opt in (1, 2), keep_baud=opt == 2)
        if s.v("id") != old_id and s.v("id") in self.servos:
            s.set(
                "id", old_id
            )  # factory id taken: keep ours (a real bus would collide)
            self.errors.append("factory reset: id taken, id kept")
        self._reboot_state(s)
        if s.v("id") != s.id:
            self.servos[s.v("id")] = self.servos.pop(s.id)
            s.id = s.v("id")

    def _reboot(self, s):
        s.reset(ram_only=True)
        self._reboot_state(s)

    def _reboot_state(self, s):
        self._push(s, {m[0] for m in _MAP})
        self._set(s, "torque_enable", 0.0)
        s.set("goal_position", self._pos(s, self._q(s, "counts")))

    def _clear(self, s):  # Present Position back to 0..cpr-1 (multi-turn reset)
        s.shift = 0
        s.shift = (
            -((round(self._q(s, "counts")) + self._frame(s)) // self.cpr) * self.cpr
        )

    # ---- register file <-> simulated device ------------------------------------------------------------------
    def _set(self, s, attr, value):
        r = self.sim.call(
            {
                "source": "dynamixel",
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

    def _frame(self, s):  # sim counts -> register frame: present = counts + frame
        return s.v("homing_offset") + s.shift

    def _pos(self, s, counts):  # present position register value
        x = round(counts) + self._frame(s)
        return (
            x % self.cpr
            if s.v("operating_mode") == 3
            else max(-(1 << 31), min(x, (1 << 31) - 1))
        )

    def _goal_counts(
        self, s, g
    ):  # goal register -> sim counts. Mode 3 stays in the present turn (no wrap crossing)
        x = round(self._q(s, "counts")) + self._frame(s)
        base = x - x % self.cpr if s.v("operating_mode") == 3 else 0
        return base + g - self._frame(s)

    def _refresh(self, s):
        """Fill the present-value registers (70, 120..146) from the sim."""
        q, cl = (
            self._q,
            lambda x, n: max(
                -(1 << (8 * n - 1)), min(round(x), (1 << (8 * n - 1)) - 1)
            ),
        )
        counts, per_rad, on = q(s, "counts"), self.cpr / (2 * pi), s.mem[64] != 0
        vel = q(s, "velocity") / VEL_UNIT
        goal_c, setp_c = q(s, "goal_position") * per_rad, q(s, "setpoint") * per_rad
        ongoing = on and q(s, "goal_speed") > 0 and abs(goal_c - setp_c) > 0.5
        s.set("hardware_error_status", s.cur)
        s.set("realtime_tick", self.sim.tick % 32768)
        s.set("moving", int(abs(vel) > s.v("moving_threshold") or ongoing))
        s.set(
            "moving_status",
            int(on and abs(goal_c - counts) <= 2) | ongoing << 1 | ongoing << 4,
        )
        s.set("present_pwm", cl(q(s, "load") * PWM_FULL, 2))
        if "present_current" in s.regs:  # XL330: mA; XL430: 0.1 % load
            s.set("present_current", cl(q(s, "current") * 1000, 2))
        else:
            s.set("present_load", cl(q(s, "load") * 1000, 2))
        s.set("present_velocity", cl(vel, 4))
        s.set("present_position", self._pos(s, counts))
        s.set("position_trajectory", self._pos(s, setp_c))
        s.set("present_input_voltage", round(q(s, "voltage") * 10))
        s.set("present_temperature", cl(q(s, "temperature"), 1))
