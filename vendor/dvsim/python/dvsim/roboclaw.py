"""Virtual Basicmicro RoboClaw Packet Serial controller: a pty speaking the RoboClaw protocol, backed by two simulated servo devices
(src/mujoco/devices/servo.cpp, firmware modes 1 duty / 2 velocity PID, see the Servo notes there).

    port = RoboclawPort(sim, motors={1: "/devices/wheel_l", 2: "/devices/wheel_r"}, address=0x80, baud=38400)
    port.device_path        # e.g. /dev/pts/7: open it with the unmodified `basicmicro` library (PyPI basicmicro, MIT) or any packet client

Sources (fetched 2026-10-07): RoboClaw Series User Manual rev 5.5 (https://www.electrokit.com/upload/product/41015/41015436/roboclaw_user_manual.pdf: "Packet
Serial Mode", commands 0-7, 16-24, 28-40, 48-49, 52-56, 90, 133-136), the official library https://github.com/basicmicro/basicmicro_python
(v2.0.10, which is what the wire format below was checked against), digest: docs/research/rover.md.

Frame: a write is `ADDR CMD DATA.. CRC16`, a read request is just `ADDR CMD` (no CRC), ADDR 0x80..0x87 (this port answers one), CRC-16/XMODEM (poly 0x1021, init 0, MSB first) over ADDR CMD DATA,
big-endian, multi-byte values big-endian. A write command is answered by 0xFF; a read command by `DATA.. CRC16` where the CRC covers
ADDR, CMD and the reply data. A packet with a bad CRC, a wrong address or an unknown command gets NO reply (manual: "If the packet was not
valid for any reason no acknowledgement will be sent back") and the receive buffer is cleared (a real unit discards the packet on the
10 ms inter-byte timeout; the bytes behind a bad packet are lost with it). A partial packet is dropped after 10 ms (manual).

Implemented (S = the sim, P = stored in the port only):
  0 / 1 / 4 / 5   drive M1 / M2 forward / backward, 7 bit (0..127 = 0..100 %) ............ S  duty, no ramp
  6 / 7           drive M1 / M2, 7 bit (0 full reverse, 64 stop, 127 full forward) ....... S  duty
  16 / 17         read encoder count (4, unsigned 32 bit) + status byte .................. S  `counts` - the reset offset (bit0 underflow,
                                                                                              bit1 backwards, bit2 overflow, the two flags
                                                                                              clear after reading)
  18 / 19         read encoder speed (4, pulses / s, magnitude) + direction byte ......... S  `speed`
  20              reset both encoder counters to 0 ....................................... P  offset
  22 / 23         set encoder 1 / 2 count ................................................ P  offset
  21              firmware version string ................................................ P  placeholder text
  24              main battery voltage, 0.1 V ............................................ S  `voltage`
  28 / 29         velocity PID: D, P, I (4 bytes each, /65536) and QPPS .................. S  vel_d / vel_p / vel_i / qpps
  55 / 56         read P, I, D, QPPS ...................................................... S
  32 / 33 / 34    drive M1 / M2 / both with signed duty (+-32767) ......................... S  duty, no ramp
  35 / 36 / 37    drive with signed speed (counts / s), velocity PID, no ramp ............. S  mode 2
  38 / 39 / 40    the same with acceleration (counts / s^2, shared by both in 40) ......... S  vel_accel
  52 / 53 / 54    duty + acceleration: duty (2), accel (4) per motor (the library sends 4 bytes for 52 / 53 as well; the manual says 2:
                  deviation, library followed). accel / 32767 = duty / s ................... S  duty_accel
  48              motor PWM, +-32767 ...................................................... S  `duty`
  49              motor current, signed 10 mA ............................................. S  `current`
  90              status, 4 bytes (library reads a long; manual rev 5.5 bit layout: 0x0001 / 0x0002 M1 / M2 overcurrent warning when
                  |current| exceeds the stored max current (133 / 134; 0 = off), 0x0800 main battery low, 0x0400 main battery high
                  against the device's min / max voltage) ................................. S
  133 / 134 / 135 / 136  set / read max current (4 + 4 bytes, 10 mA) ..................... P  stored, NOT applied to the motor
NOT implemented (no reply): mixed-mode 8-13, buffered / position commands (41-51, 61-67), timeout, logic battery, temperatures, EEPROM,
modes, 78-79, PWM mode. The velocity PID and the 20 ms speed filter inside the Servo are estimates (the manual gives only the
defaults P = 0x10000, I = 0x8000, D = 0x4000, QPPS = 44000 and "QPPS = speed at 100 % power"): docs/research/rover.md.

Every write is a SetAttr on the servo device with source "roboclaw", so a Host log replays it exactly (the pty traffic and its wall-clock
timing are not part of the replay). Timing: with `realtime=True` each request byte and each reply byte takes 10 / baud s.
"""

import os
import select
import struct
import termios
import threading
import time
import tty

ACK = b"\xff"
# command -> bytes of data after the command byte, writes (reply 0xFF) and reads (reply data + CRC)
WRITES = {
    0: 1,
    1: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    20: 0,
    22: 4,
    23: 4,
    28: 16,
    29: 16,
    32: 2,
    33: 2,
    34: 4,
    35: 4,
    36: 4,
    37: 8,
}
WRITES |= {38: 8, 39: 8, 40: 12, 52: 6, 53: 6, 54: 12, 133: 8, 134: 8}
READS = {16, 17, 18, 19, 21, 24, 48, 49, 55, 56, 90, 135, 136}
VERSION = b"USB Roboclaw 2x7A v4.1.12 (dvsim sim)\n\x00"  # placeholder text; the manual's example is "RoboClaw 10.2A v4.1.11"
# defaults of command 28 (manual): QPPS 44000, P 0x10000, I 0x8000, D 0x4000 (register / 65536)
DEFAULTS = {"qpps": 44000.0, "vel_p": 1.0, "vel_i": 0.5, "vel_d": 0.25}


def crc16(data: bytes, crc: int = 0) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1) & 0xFFFF
    return crc


def packet(addr: int, cmd: int, data: bytes = b"") -> bytes:
    p = bytes([addr, cmd]) + data
    return p + struct.pack(">H", crc16(p))


class RoboclawPort:
    def __init__(
        self,
        sim,
        motors: dict,
        address: int = 0x80,
        baud: int = 38400,
        symlink=None,
        realtime: bool = True,
    ):
        """sim: a Sim loaded with host=True. motors: {1: "/devices/<servo>", 2: ...} (M1, M2; a missing one reads zeros).
        address: 0x80..0x87. Starts the motors in duty mode, torque on, duty 0, the manual's default PID / QPPS."""
        if not 0x80 <= address <= 0x87:
            raise ValueError("RoboClaw packet address is 0x80..0x87")
        if not motors or not set(motors) <= {1, 2}:
            raise ValueError("motors needs keys 1 and / or 2")
        self.sim, self.motors, self.address, self.baud, self.realtime = (
            sim,
            dict(motors),
            address,
            baud,
            realtime,
        )
        self.errors = []  # rejected SetAttrs and dropped packets, for tests / diagnostics
        self.drop = (
            False  # comms fault: while True every byte the host sends is discarded
        )
        self.offset = dict.fromkeys(motors, 0.0)
        self.max_current = dict.fromkeys(motors, 0)
        self._last = {
            m: (0, False) for m in motors
        }  # last encoder read / reset: (raw count, backwards)
        self._flags = dict.fromkeys(motors, 0)
        for m in motors:
            self._set(m, "torque_enable", 1)
            self._set(m, "mode", 1)
            self._set(m, "goal_duty", 0)
            for k, v in DEFAULTS.items():
                self._set(m, k, v)
        self._rx = bytearray()
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
            target=self._serve, name="roboclaw-port", daemon=True
        )
        self._thread.start()

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

    # ---- I/O thread ----------------------------------------------------------------------------------------------
    def _wire(self, nbytes):
        if self.realtime:
            time.sleep(nbytes * 10 / self.baud)

    def _serve(self):
        while not self._stop.is_set():
            r, _, _ = select.select([self._master], [], [], 0.01)
            if not r:
                self._rx.clear()  # 10 ms without a byte: a partial packet is discarded
                continue
            try:
                data = os.read(self._master, 4096)
            except (BlockingIOError, InterruptedError):
                continue
            if self.drop:
                self._rx.clear()
                continue
            for n, cmd, body in self.feed(data):
                self._wire(
                    n
                )  # the request's time on the wire; state is sampled after it
                reply = self.handle(cmd, body)
                if reply:
                    self._wire(len(reply))
                    try:
                        os.write(self._master, reply)
                    except BlockingIOError:
                        self.errors.append("tx buffer full: reply dropped")

    def feed(self, data: bytes) -> list:
        """Append raw bytes, return the complete valid request packets as (wire length, command, data)."""
        rx, out = self._rx, []
        rx += data
        while len(rx) >= 2:
            cmd = rx[1]
            if rx[0] != self.address or (cmd not in WRITES and cmd not in READS):
                self.errors.append(
                    f"ignored packet {bytes(rx[:2]).hex()}"
                )  # not ours / unknown: discard the lot
                rx.clear()
                break
            n = (
                2 if cmd in READS else 2 + WRITES[cmd] + 2
            )  # a read request is [ADDR, CMD] only, no CRC (manual)
            if len(rx) < n:
                break
            pkt = bytes(rx[:n])
            del rx[:n]
            if cmd in READS:
                out.append((n, cmd, b""))
                continue
            if crc16(pkt[:-2]) != struct.unpack(">H", pkt[-2:])[0]:
                self.errors.append(f"bad crc, cmd {cmd}")
                rx.clear()
                break
            out.append((n, cmd, pkt[2:-2]))
        return out

    # ---- commands ------------------------------------------------------------------------------------------------
    def handle(self, cmd: int, d: bytes) -> bytes:
        """One valid request -> the reply bytes (b"" for none)."""
        if cmd in READS:
            return packet(self.address, cmd, self._read(cmd))[
                2:
            ]  # the CRC covers ADDR CMD DATA, only DATA CRC goes out
        self._write(cmd, d)
        return ACK

    def _write(self, cmd, d):
        if cmd in (0, 1, 4, 5):
            v = d[0] / 127 * (-1 if cmd in (1, 5) else 1)
            return self._duty((1,) if cmd < 4 else (2,), v, 0)
        if cmd in (6, 7):
            return self._duty(
                (1,) if cmd == 6 else (2,), (d[0] - 64) / (63 if d[0] >= 64 else 64), 0
            )
        if cmd == 20:
            self.offset = {m: self._q(m, "counts") for m in self.motors}
            self._last = {m: (0, False) for m in self.motors}
        elif cmd in (22, 23):
            m = cmd - 21
            if m in self.motors:
                v = struct.unpack(">I", d)[0]
                self.offset[m] = self._q(m, "counts") - v
                self._last[m] = (v, False)
        elif cmd in (28, 29):
            dd, p, i, q = struct.unpack(">4I", d)
            for k, v in (
                ("vel_d", dd / 65536),
                ("vel_p", p / 65536),
                ("vel_i", i / 65536),
                ("qpps", q),
            ):
                self._set(cmd - 27, k, v)
        elif cmd in (32, 33):
            self._duty((cmd - 31,), struct.unpack(">h", d)[0] / 32767, 0)
        elif cmd == 34:
            a, b = struct.unpack(">hh", d)
            self._duty((1,), a / 32767, 0)
            self._duty((2,), b / 32767, 0)
        elif cmd in (35, 36, 37):
            for m, v in zip(
                (1, 2) if cmd == 37 else (cmd - 34,),
                struct.unpack(">" + "i" * (len(d) // 4), d),
            ):
                self._speed(m, v, 0)
        elif cmd in (38, 39, 40):
            acc, *v = struct.unpack(">I" + "i" * ((len(d) - 4) // 4), d)
            for m, s in zip((1, 2) if cmd == 40 else (cmd - 37,), v):
                self._speed(m, s, acc)
        elif cmd in (52, 53):
            duty, acc = struct.unpack(">hI", d)
            self._duty((cmd - 51,), duty / 32767, acc / 32767)
        elif cmd == 54:
            d1, a1, d2, a2 = struct.unpack(">hIhI", d)
            self._duty((1,), d1 / 32767, a1 / 32767)
            self._duty((2,), d2 / 32767, a2 / 32767)
        elif cmd in (133, 134):
            if cmd - 132 in self.max_current:
                self.max_current[cmd - 132] = struct.unpack(">II", d)[0]

    def _read(self, cmd) -> bytes:
        if cmd in (16, 17):
            m = cmd - 15
            raw = (
                round(self._q(m, "counts") - self.offset.get(m, 0))
                if m in self.motors
                else 0
            )
            prev = self._last.get(m)
            if prev:
                if prev[0] >= 0 > raw:
                    self._flags[m] |= 1  # underflow: below 0
                elif prev[0] < 0 <= raw:
                    self._flags[m] |= 4  # overflow: back above 2^32 - 1 -> 0
            back = (
                raw < prev[0]
                if prev and raw != prev[0]
                else (prev[1] if prev else False)
            )
            self._last[m] = (raw, back)
            st = self._flags.get(m, 0) | (2 if back else 0)
            self._flags[m] = 0
            return struct.pack(">IB", raw & 0xFFFFFFFF, st)
        if cmd in (18, 19):
            s = round(self._q(cmd - 17, "speed")) if cmd - 17 in self.motors else 0
            return struct.pack(">IB", abs(s), s < 0)
        if cmd == 21:
            return VERSION
        if cmd == 24:
            return struct.pack(">H", round(self._q(min(self.motors), "voltage") * 10))
        if cmd == 48:
            return struct.pack(
                ">hh",
                *(
                    round(self._q(m, "duty") * 32767) if m in self.motors else 0
                    for m in (1, 2)
                ),
            )
        if cmd == 49:
            return struct.pack(
                ">hh",
                *(
                    round(self._q(m, "current") * 100) if m in self.motors else 0
                    for m in (1, 2)
                ),
            )
        if cmd in (55, 56):
            m = cmd - 54
            q = lambda a: self._q(m, a) if m in self.motors else 0
            return struct.pack(
                ">4I",
                round(q("vel_p") * 65536),
                round(q("vel_i") * 65536),
                round(q("vel_d") * 65536),
                round(q("qpps")),
            )
        if cmd == 90:
            m0 = min(self.motors)
            v, st = self._q(m0, "voltage"), 0
            for m in self.motors:
                if (
                    self.max_current[m]
                    and abs(self._q(m, "current")) * 100 > self.max_current[m]
                ):
                    st |= m  # 0x0001 M1, 0x0002 M2 overcurrent warning
            if v < self._q(m0, "min_voltage"):
                st |= 0x0800
            if v > self._q(m0, "max_voltage"):
                st |= 0x0400
            return struct.pack(">I", st)
        m = cmd - 134  # 135 / 136: max current, min current is always 0
        return struct.pack(">II", self.max_current.get(m, 0), 0)

    # ---- sim ------------------------------------------------------------------------------------------------------
    def _duty(self, ms, duty, accel):
        for m in ms:
            self._set(m, "mode", 1)
            self._set(m, "duty_accel", accel)
            self._set(m, "goal_duty", max(-1.0, min(1.0, duty)))

    def _speed(self, m, v, accel):
        self._set(m, "mode", 2)
        self._set(m, "vel_accel", accel)
        self._set(m, "goal_velocity", v)

    def _set(self, m, attr, value):
        if m not in self.motors:
            return
        r = self.sim.call(
            {
                "source": "roboclaw",
                "cmd_type": "SetAttr",
                "cmd": {
                    "path": self.motors[m],
                    "attr": attr,
                    "frame": "",
                    "values": [float(value)],
                },
            }
        )
        if not r["ok"]:
            self.errors.append(f"{self.motors[m]}.{attr}: {r['message']}")

    def _q(self, m, attr):
        return float(self.sim.query(self.motors[m], attr)[0][0])
