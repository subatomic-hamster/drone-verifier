// M10 exit checks: the viewer under a display (xvfb-run) with a Vulkan device (lavapipe / RADV / ANV).
//   - scripted 120 frames: ticks advance, the viewer's commands are in the replay log with source "viewer"
//   - the log replays (in process and with dvsim-replay) to the live run's hash
//   - the same scripted commands without any viewer give the same hash: the viewer is a pure reader
//   - error paths: no DISPLAY / bad model / bad script exit with a clear message, not a crash
#include <gtest/gtest.h>

#include <sys/wait.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "core/host.h"
#include "mujoco/mujoco_solver.h"
#include "dvsim_generated.h"
#include "viewer/design.h"
#include "viewer/viewer.h"

namespace {
const std::string kData = DVSIM_TEST_DATA_DIR;
const std::string kModel = kData + "/pendulum.xml";

// Free-running (rate 0, then rate 1) plus Step / SetAttr: the number of ticks between commands depends on timing, so
// only the replay (which records applied ticks) is expected to reproduce it. `select` exercises the inspector's reads.
const char* kScriptRun =
    "2 select /world/pole/hinge; 5 run 0; 40 pause; 50 step 5; 60 setattr /world/pole/hinge qpos 0.3; 80 run 1; 110 pause";
// Only commands whose effect does not depend on timing: live, replayed and viewer-less runs must all agree bit for bit.
const char* kScriptDet =
    "2 select /world/pole/hinge; 3 step 5; 10 setattr /world/pole/hinge qpos 0.4; 12 select /world/pole; 20 step 20; "
    "30 setattr /actuators/torque ctrl 1.5; 40 step 100; 60 step 7; 70 setattr /world/pole/hinge qvel 0.5; 80 step 3";

Session session() {
  Session s;
  s.caps = {"read", "control"};
  s.root = kData;
  return s;
}
World pendulum() { return World(load_mujoco, kModel); }

bool have_display() { return std::getenv("DISPLAY") && *std::getenv("DISPLAY"); }
#define NEED_DISPLAY() \
  if (!have_display()) GTEST_SKIP() << "needs a display: run under xvfb-run"

struct Rec {
  std::uint64_t applied_tick;
  std::string source;
  dvsim::Command type;
};
template <class T>
T rd(const std::vector<std::byte>& b, std::size_t& pos) {
  T v;
  std::memcpy(&v, b.data() + pos, sizeof v);
  pos += sizeof v;
  return v;
}
// Walks the documented log layout (core/replay.h) and decodes every record's envelope.
std::vector<Rec> records(const std::vector<std::byte>& log) {
  std::size_t pos = 8 + 4 + 8;  // magic, format version, schema hash
  pos += rd<std::uint32_t>(log, pos);  // solver version (rd advanced pos past the length)
  pos += rd<std::uint64_t>(log, pos);  // model
  pos += rd<std::uint64_t>(log, pos);  // initial state
  std::vector<Rec> out;
  const std::size_t end = log.size() - kReplayFooterSize;
  while (pos < end) {
    const auto tick = rd<std::uint64_t>(log, pos);
    rd<std::uint8_t>(log, pos);  // expected error
    const auto n = rd<std::uint32_t>(log, pos);
    const dvsim::Envelope& e = *dvsim::GetEnvelope(reinterpret_cast<const std::uint8_t*>(log.data() + pos));
    out.push_back({tick, e.source() ? e.source()->str() : "", e.cmd_type()});
    pos += n;
  }
  EXPECT_EQ(pos, end);
  return out;
}

struct Outcome {
  std::uint64_t tick;
  std::uint64_t hash;
  std::vector<std::byte> log;
};
Outcome finish(Host& h) {
  h.stop();
  return {h.world().tick(), h.world().state_hash(), h.world().log()->bytes()};
}
Outcome with_viewer(const std::string& script, int frames) {
  Host h(pendulum(), session());
  ViewerOptions o;
  o.frames = frames;
  o.script = script;
  o.width = 640;
  o.height = 480;
  run_viewer(h, o);
  return finish(h);
}
// The same scripted commands, submitted in order straight to a Host: no window, no viewer.
Outcome without_viewer(const std::string& script) {
  Host h(pendulum(), session());
  for (const ScriptCmd& c : parse_script(script))
    if (c.action != "select") h.call(script_envelope(c, "viewer"));
  return finish(h);
}
std::string hex(std::uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

struct Out {
  int code;
  std::string text;  // stdout + stderr
};
Out shell(const std::string& cmd) {
  Out o{-1, ""};
  FILE* p = popen((cmd + " 2>&1").c_str(), "r");
  if (!p) return o;
  char buf[512];
  while (std::fgets(buf, sizeof buf, p)) o.text += buf;
  const int st = pclose(p);
  o.code = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);  // >= 128: killed by a signal (a crash)
  return o;
}
std::string field(const std::string& text, const std::string& key) {  // "key=value" token
  const auto at = text.find(key + "=");
  if (at == std::string::npos) return "";
  const auto b = at + key.size() + 1;
  return text.substr(b, text.find_first_of(" \n", b) - b);
}
}  // namespace

TEST(viewer_smoke, scripted_120_frames_logged_as_viewer_and_replays_to_live_hash) {
  NEED_DISPLAY();
  const Outcome live = with_viewer(kScriptRun, 120);
  const std::vector<Rec> recs = records(live.log);
  std::cout << "[ viewer ] 120 frames: ticks=" << live.tick << " records=" << recs.size() << " live hash=" << hex(live.hash) << "\n";
  EXPECT_GT(live.tick, 100u) << "ticks must advance while the viewer runs the sim";

  // Run, Pause, Step, SetAttr, Run, Pause: all state-changing commands came from the viewer; its reads are not logged.
  const std::vector<dvsim::Command> want = {dvsim::Command::Run,     dvsim::Command::Pause, dvsim::Command::Step,
                                           dvsim::Command::SetAttr, dvsim::Command::Run,   dvsim::Command::Pause};
  ASSERT_EQ(recs.size(), want.size());
  std::size_t viewer_records = 0;
  for (std::size_t i = 0; i < recs.size(); ++i) {
    EXPECT_EQ(recs[i].type, want[i]) << "record " << i;
    if (recs[i].source == "viewer") ++viewer_records;
    if (i) EXPECT_GE(recs[i].applied_tick, recs[i - 1].applied_tick);
  }
  EXPECT_EQ(viewer_records, recs.size());
  std::cout << "[ viewer ] source=viewer records: " << viewer_records << "/" << recs.size() << "\n";

  const World r = replay(live.log, load_mujoco_bytes, load_mujoco, kData);  // no viewer, no Host thread
  std::cout << "[ viewer ] replay (no viewer): tick=" << r.tick() << " hash=" << hex(r.state_hash()) << "\n";
  EXPECT_EQ(r.tick(), live.tick);
  EXPECT_EQ(r.state_hash(), live.hash);
}

TEST(viewer_smoke, same_commands_without_viewer_give_the_same_hash) {
  NEED_DISPLAY();
  const Outcome live = with_viewer(kScriptDet, 120);
  const Outcome bare = without_viewer(kScriptDet);
  const World r = replay(live.log, load_mujoco_bytes, load_mujoco, kData);
  std::cout << "[ viewer ] deterministic script: live=" << hex(live.hash) << " (tick " << live.tick << ")  no-viewer=" << hex(bare.hash)
            << " (tick " << bare.tick << ")  replay=" << hex(r.state_hash()) << "\n";
  EXPECT_EQ(live.tick, 135u);  // 5 + 20 + 100 + 7 + 3
  EXPECT_EQ(bare.tick, live.tick);
  EXPECT_EQ(bare.hash, live.hash);
  EXPECT_EQ(r.state_hash(), live.hash);
  EXPECT_EQ(records(live.log).size(), 8u);  // 5 steps + 3 setattrs; the inspector's queries are reads
}

TEST(viewer_smoke, dvsim_view_cli_log_replays_with_dvsim_replay) {
  NEED_DISPLAY();
  std::filesystem::create_directories(DVSIM_VIEWER_OUT_DIR);
  const std::string log = std::string(DVSIM_VIEWER_OUT_DIR) + "/viewer_smoke.nlog";
  const Out v = shell(std::string(DVSIM_VIEW_EXE) + " " + kModel + " --frames 120 --script '" + kScriptRun + "' --log " + log);
  ASSERT_EQ(v.code, 0) << v.text;
  const Out r = shell(std::string(DVSIM_CLI_EXE) + " " + log);
  ASSERT_EQ(r.code, 0) << r.text;
  std::cout << "[ viewer ] dvsim-view : " << v.text << "[ viewer ] dvsim-replay: " << r.text;
  EXPECT_EQ(field(v.text, "state_hash"), field(r.text, "state_hash"));
  EXPECT_EQ(field(v.text, "tick"), field(r.text, "tick"));
  EXPECT_FALSE(field(v.text, "state_hash").empty());
}

// Not window tests: they must hold with or without a display, so they never skip.
TEST(viewer_smoke, no_display_is_a_clear_error_not_a_crash) {
  const Out o = shell(std::string("env -u DISPLAY -u WAYLAND_DISPLAY ") + DVSIM_VIEW_EXE + " " + kModel + " --frames 3");
  std::cout << "[ viewer ] no DISPLAY: exit " << o.code << ": " << o.text;
  EXPECT_EQ(o.code, 1);
  EXPECT_NE(o.text.find("dvsim-view: error: cannot open a window"), std::string::npos) << o.text;
  EXPECT_NE(o.text.find("DISPLAY"), std::string::npos) << o.text;
}

// Off-screen viewer: no display needed (run with DISPLAY unset to prove it); needs a Vulkan device.
TEST(viewer_offscreen, frames_are_read_back_and_the_sim_hash_matches_a_viewerless_run) {
  Host h(pendulum(), session());
  ViewerOptions o;
  o.frames = 120;  // the script's last command is at frame 80
  o.script = kScriptDet;
  o.width = 320;
  o.height = 200;
  o.offscreen = true;
  run_viewer(h, o);
  const Outcome live = finish(h);
  std::vector<std::uint8_t> px;
  int w = 0, hh = 0;
  const std::uint64_t seq = viewer_latest_frame(&px, &w, &hh);
  EXPECT_GE(seq, 120u);
  ASSERT_EQ(w, 320);
  ASSERT_EQ(hh, 200);
  ASSERT_EQ(px.size(), 320u * 200u * 4u);
  std::size_t distinct = 0;  // sky + floor + pendulum + the panels: far from a single colour
  for (std::size_t i = 4; i < px.size() && distinct < 64; i += 4) distinct += std::memcmp(&px[i], &px[0], 4) != 0;
  EXPECT_GE(distinct, 64u) << "the frame is blank";
  EXPECT_EQ(px[3], 255);  // opaque
  EXPECT_EQ(live.hash, without_viewer(kScriptDet).hash);
}

TEST(viewer_offscreen, dvsim_view_cli_runs_without_a_display_and_names_its_device) {
  const Out v = shell(std::string("env -u DISPLAY -u WAYLAND_DISPLAY ") + DVSIM_VIEW_EXE + " " + kModel + " --frames 20 --offscreen 320x200");
  std::cout << "[ viewer ] off-screen CLI: " << v.text;
  EXPECT_EQ(v.code, 0) << v.text;
  EXPECT_NE(v.text.find("dvsim-view: off-screen 320x200 on "), std::string::npos) << v.text;
  EXPECT_FALSE(field(v.text, "state_hash").empty());
  const Out bad = shell(std::string(DVSIM_VIEW_EXE) + " " + kModel + " --offscreen 0x0");
  EXPECT_EQ(bad.code, 2) << bad.text;
}

TEST(viewer_smoke, bad_model_bad_script_and_usage_exit_cleanly) {
  const std::string exe = DVSIM_VIEW_EXE;
  Out o = shell(exe + " /no/such/model.xml --frames 1");
  EXPECT_EQ(o.code, 1) << o.text;
  EXPECT_NE(o.text.find("dvsim-view: error:"), std::string::npos) << o.text;
  o = shell(exe + " " + kModel + " --script '5 bogus'");  // rejected before any window exists
  EXPECT_EQ(o.code, 1) << o.text;
  EXPECT_NE(o.text.find("unknown action 'bogus'"), std::string::npos) << o.text;
  o = shell(exe);
  EXPECT_EQ(o.code, 2) << o.text;
  EXPECT_NE(o.text.find("usage:"), std::string::npos) << o.text;
}

TEST(viewer_smoke, script_parser_and_envelopes) {
  const std::vector<ScriptCmd> s = parse_script("5 run 0.5; 7 setattr /a/b qpos 1 2; 9 pause;");
  ASSERT_EQ(s.size(), 3u);
  EXPECT_EQ(s[1].frame, 7);
  EXPECT_EQ(s[1].args.size(), 4u);
  const std::vector<std::uint8_t> e = script_envelope(s[0], "viewer");
  const dvsim::Envelope& env = *dvsim::GetEnvelope(e.data());
  EXPECT_EQ(env.cmd_type(), dvsim::Command::Run);
  EXPECT_DOUBLE_EQ(env.cmd_as_Run()->rate(), 0.5);
  EXPECT_EQ(env.source()->str(), "viewer");
  EXPECT_THROW(parse_script("x run 1"), std::runtime_error);
  EXPECT_THROW(parse_script("1 step"), std::runtime_error);
  EXPECT_THROW(parse_script("1 setattr /a b"), std::runtime_error);
  EXPECT_EQ(parse_script("3 inst arm").at(0).action, "inst");
  EXPECT_THROW(parse_script("3 inst"), std::runtime_error);
}

// ---- Design window (report of a `dvsim design build`) -------------------------------------------------------------------
const std::string kReport = kData + "/design_demo.report.json";

TEST(viewer_design, report_parses_and_instances_resolve_to_body_prims) {  // no window needed
  const design::Report r = design::load_report(kReport);
  EXPECT_EQ(r.assembly, "pendulum_demo");
  EXPECT_NEAR(r.mass_total, 1.0025, 1e-12);
  ASSERT_EQ(r.instances.size(), 2u);
  EXPECT_EQ(r.instances[0].part, "demo-rod");
  EXPECT_EQ(r.instances[0].body, "pole");
  EXPECT_NEAR(r.instances[1].mass, 2.5e-3, 1e-15);
  ASSERT_EQ(r.by_group.size(), 1u);
  EXPECT_EQ(r.by_group[0].first, "frame");
  ASSERT_EQ(r.targets.size(), 1u);
  EXPECT_FALSE(r.targets[0].within);
  ASSERT_EQ(r.findings.size(), 3u);
  EXPECT_EQ(r.findings[0].severity, "error");
  EXPECT_EQ(r.findings[0].code, "DEMO_ERROR");
  EXPECT_NE(r.findings[0].message.find("\xC3\xA9"), std::string::npos) << "\\u00e9 decodes to UTF-8";
  EXPECT_EQ(r.findings[0].where, (std::vector<std::string>{"port x", "arm"}));
  const std::vector<std::string> prims = {"/world", "/world/base", "/world/base/pole", "/world/base/pole/hinge"};
  EXPECT_EQ(design::body_path(prims, "pole"), "/world/base/pole");
  EXPECT_EQ(design::body_path(prims, "ole"), "");
  EXPECT_EQ(design::body_path(prims, "ghost"), "");
  EXPECT_THROW(design::parse_report("{\"assembly\": "), std::runtime_error);
  EXPECT_THROW(design::load_report("/no/such/report.json"), std::runtime_error);
}

TEST(viewer_design, window_opens_with_a_report_and_instance_clicks_do_not_touch_the_sim) {
  NEED_DISPLAY();
  const std::string script = "3 inst arm; 10 inst ghost; 20 inst nothing; 30 step 4";
  Host h(pendulum(), session());
  ViewerOptions o;
  o.frames = 60;
  o.script = script;
  o.design_report = kReport;
  run_viewer(h, o);
  const Outcome with = finish(h);
  Host bare(pendulum(), session());
  bare.call(script_envelope(parse_script("0 step 4").at(0), "viewer"));
  const Outcome ref = finish(bare);
  std::cout << "[ viewer ] design window, 60 frames: tick=" << with.tick << " hash=" << hex(with.hash) << "\n";
  EXPECT_EQ(with.tick, 4u);  // only the Step ran: selecting instances is UI state, not a bus command
  EXPECT_EQ(with.hash, ref.hash);
  EXPECT_EQ(records(with.log).size(), 1u);
}

TEST(viewer_design, dvsim_view_finds_the_report_beside_the_model_and_rejects_a_bad_one) {
  NEED_DISPLAY();
  const Out ok = shell(std::string(DVSIM_VIEW_EXE) + " " + kModel + " --frames 30 --design " + kReport + " --script '2 inst arm'");
  EXPECT_EQ(ok.code, 0) << ok.text;
  const Out bad = shell(std::string(DVSIM_VIEW_EXE) + " " + kModel + " --frames 3 --design /no/such.report.json");
  EXPECT_EQ(bad.code, 1) << bad.text;
  EXPECT_NE(bad.text.find("design report: cannot read"), std::string::npos) << bad.text;
}

// ---- Design editing: the window sends JSON requests to ViewerOptions::design_api and loads what it answers ------------------
TEST(viewer_design, edit_script_actions_parse_and_need_an_editor) {  // no window needed
  EXPECT_EQ(parse_script("1 dsnap p a.b id").at(0).args.size(), 3u);
  EXPECT_EQ(parse_script("1 dautowire_all; 2 dundo").size(), 2u);
  EXPECT_THROW(parse_script("1 dsnap p"), std::runtime_error);
  EXPECT_THROW(parse_script("1 dundo x"), std::runtime_error);
  const Out o = shell(std::string(DVSIM_VIEW_EXE) + " " + kModel + " --frames 1 --script '1 dundo'");
  EXPECT_EQ(o.code, 1) << o.text;
  EXPECT_NE(o.text.find("needs a design editor"), std::string::npos) << o.text;
  const design::State s = design::state_from(design::Parser(R"({"rev":"ab","xml":"/m.xml","undo":2,"redo":1,
      "report":{"assembly":"a","mass":{"total":1.5,"by_group":{"g":1.5}},"instances":[{"id":"i","part":"p","body":"i","mass":1.5}],
        "targets":[],"findings":[{"code":"C","severity":"error","message":"m","fix":"f","where":["w.x"],"instances":["i"]}]},
      "ports":{"i":[{"name":"p1","kind":"mech","type":"t","used":true}]}})").parse());
  EXPECT_EQ(s.rev, "ab");
  EXPECT_EQ(s.undo, 2);
  ASSERT_EQ(s.report.findings.size(), 1u);
  EXPECT_EQ(s.report.findings[0].instances, (std::vector<std::string>{"i"}));
  EXPECT_TRUE(s.ports.at("i").at(0).used);
  EXPECT_EQ(design::request("design_edit", "{\"ops\":[]}", true), R"({"tool":"design_edit","args":{"ops":[]},"replace_last":true})");
  EXPECT_EQ(design::quote("a\"b\\c"), R"("a\"b\\c")");
}

TEST(viewer_design, scripted_edits_become_ordered_requests_and_a_new_rev_reloads_the_model) {
  NEED_DISPLAY();
  std::vector<std::string> seen;
  int rev = 0;  // bumped by every edit the fake backend "applies"
  const auto state = [&] {
    return R"("state":{"rev":"r)" + std::to_string(rev) + R"(","xml":")" + kModel + R"(","build_error":"","path":"","undo":1,"redo":0,
      "report":{"assembly":"pendulum_demo","mass":{"total":1,"by_group":{}},"instances":[{"id":"arm","part":"demo-rod","group":"frame","body":"pole","mass":1}],
        "targets":[],"findings":[{"code":"X","severity":"warning","message":"m","fix":"","where":["arm"],"instances":["arm"]}]},
      "ports":{"arm":[{"name":"p","kind":"mech","type":"t","used":false}]}})";
  };
  Host h(pendulum(), session());
  ViewerOptions o;
  o.frames = 40;
  o.width = 320;
  o.height = 240;
  o.script = "2 dremove arm; 3 dsnap demo-rod arm.p arm2; 4 dcycle 0; 5 dundo; 6 dfinding 0; 7 dsync";
  o.design_api = [&](const std::string& req) {
    const design::Json j = design::Parser(req).parse();
    const std::string tool = j.s("tool");
    seen.push_back(req);
    if (tool == "rev") return R"({"ok":true,"result":{"rev":"r)" + std::to_string(rev) + R"("}})";
    if (tool.rfind("design_", 0) == 0) ++rev;  // library.* / ui_poll / select / state never change the model
    const std::string drc = R"("drc":{"before":{"error":0,"warning":0},"after":{"error":0,"warning":1},"new":[],"resolved":[]})";
    const std::string add = R"({"op":"add","id":"arm2","part":"demo-rod","mate":{"a":"arm.p","b":"arm2.b"},
      "snap":{"k":0,"score":100,"why":[],"interferes":[],"alternatives":[{"to":"arm.p","port":"b","k":1,"score":90,"why":[],"interferes":[]}]}})";
    return R"({"ok":true,"result":{"changes":[)" + (tool == "design_edit" && j.at("args").at("ops").arr.at(0).s("op") == "add" ? add : R"({"op":"remove","id":"arm"})") +
           "]," + drc + "}," + state() + "}";
  };
  run_viewer(h, o);
  const Outcome out = finish(h);
  std::vector<std::string> tools;
  for (const std::string& r : seen) {
    const design::Json j = design::Parser(r).parse();
    const std::string t = j.s("tool");  // the window's own housekeeping (library browser, event poll, selection mirror) is not an edit
    if (t != "rev" && t != "ui_poll" && t != "select" && t != "flatten.state" && t != "ui.report" && t.rfind("library.", 0) != 0) tools.push_back(j.s("tool") + (j.at("replace_last").num != 0 ? "+replace_last" : ""));
  }
  const std::vector<std::string> want = {"state", "design_edit", "design_edit", "design_edit+replace_last", "design_undo", "state"};
  EXPECT_EQ(tools, want);
  std::size_t loads = 0;
  for (const Rec& r : records(out.log))
    if (r.type == dvsim::Command::LoadModel) {
      EXPECT_EQ(r.source, "viewer");
      ++loads;
    }
  std::cout << "[ viewer ] fake design backend: requests " << tools.size() << " in order, " << loads << " model reloads (one per new revision)\n";
  EXPECT_EQ(loads, 4u);  // remove, add, cycle, undo each changed the revision; the first state and the final sync did not
}

// ---- The mode bar: BUILD / LIBRARY / WIRING / SIMULATE / PROGRAM, typed search, selection, camera presets ----------------------
TEST(viewer_design, every_mode_draws_and_ui_script_verbs_reach_the_backend_without_touching_the_sim) {
  NEED_DISPLAY();
  EXPECT_EQ(parse_script("1 mode library; 2 search motor; 3 uicam iso; 4 uiframe; 5 uisel a,b").size(), 5u);
  EXPECT_THROW(parse_script("1 mode"), std::runtime_error);
  std::vector<std::string> seen;
  Host h(pendulum(), session());
  ViewerOptions o;
  o.frames = 120;
  o.width = 640;
  o.height = 360;
  o.script = "3 mode library; 6 search motor; 30 mode wiring; 36 mode simulate; 42 mode program; 60 mode build; 64 uisel arm; 68 uicam iso; 72 uiframe; 80 step 2";
  o.design_api = [&](const std::string& req) -> std::string {
    const design::Json j = design::Parser(req).parse();
    const std::string tool = j.s("tool");
    seen.push_back(tool == "library.search" ? req : tool);
    if (tool == "ui_poll") return R"({"ok":true,"result":{"events":[]}})";
    if (tool == "library.categories") return R"({"ok":true,"result":{"categories":[{"path":"motor","count":2,"depth":0}]}})";
    if (tool == "library.search")
      return R"({"ok":true,"result":{"total":2,"parts":[{"id":"demo-rod","name":"Demo rod","vendor":"V","category":"motor","thumb":"","mass":0.1},
        {"id":"demo-box","name":"Demo box","vendor":"V","category":"motor","thumb":""}]}})";
    if (tool == "library.ranges") return R"({"ok":true,"result":{"ranges":{}}})";
    if (tool == "design_text") return R"({"ok":true,"result":{"text":"[assembly]\nname = \"demo\"\n"}})";
    return R"({"ok":true,"result":{},"state":{"rev":"r0","xml":")" + std::string(kModel) + R"(","build_error":"","path":"","undo":0,"redo":0,
      "report":{"assembly":"pendulum_demo","mass":{"total":1,"by_group":{}},"instances":[{"id":"arm","part":"demo-rod","group":"frame","body":"pole","mass":1}],
        "targets":[],"findings":[]},"ports":{"arm":[]}}})";
  };
  run_viewer(h, o);
  const Outcome out = finish(h);
  const auto count = [&](const std::string& s) { return std::count_if(seen.begin(), seen.end(), [&](const std::string& r) { return r.find(s) != std::string::npos; }); };
  EXPECT_GE(count("prog.files"), 1);                                      // the PROGRAM tab listed the program directory
  EXPECT_GE(count("term.open"), 1);                                       // ... and started its first shell
  EXPECT_GE(count("\"text\":\"motor\""), 1) << "the typed search never reached the backend";
  EXPECT_GE(count("ui_poll"), 1);
  EXPECT_EQ(out.tick, 2u);  // UI state is not a bus command: only the Step ran
  EXPECT_EQ(records(out.log).size(), 1u);
}

// ---- PROGRAM page: the code editor opens a file through prog.read, typing marks it dirty, save writes the text back ----------------
TEST(viewer_design, program_page_opens_a_file_types_and_saves_through_the_design_api) {
  NEED_DISPLAY();
  std::vector<std::string> seen;
  std::string written;
  std::vector<std::string> writes;
  int polls = 0;
  Host h(pendulum(), session());
  ViewerOptions o;
  o.frames = 1500;  // events are timed by ui_poll count (100 ms wall clock each): frames must outlast them on a fast GPU
  o.width = 960;
  o.height = 540;
  o.design_api = [&](const std::string& req) -> std::string {
    const design::Json j = design::Parser(req).parse();
    const std::string tool = j.s("tool");
    seen.push_back(tool);
    if (tool == "ui_poll") {
      ++polls;
      if (polls == 2) return R"({"ok":true,"result":{"events":[{"cmd":"mode","name":"program"}]}})";
      if (polls == 5) return R"({"ok":true,"result":{"events":[{"cmd":"prog_open","path":"plan.py","line":0,"col":0}]}})";
      if (polls == 9) return R"({"ok":true,"result":{"events":[{"cmd":"prog_type","text":"# edited\n"}]}})";
      if (polls == 12) return R"({"ok":true,"result":{"events":[{"cmd":"prog_cmd","name":"save"},{"cmd":"prog_term","text":"echo hi"}]}})";
      if (polls == 16) return R"({"ok":true,"result":{"events":[{"cmd":"prog_find","text":"edited","replace":"changed"}]}})";
      if (polls == 20) return R"({"ok":true,"result":{"events":[{"cmd":"prog_replace_all"}]}})";
      if (polls == 24) return R"({"ok":true,"result":{"events":[{"cmd":"prog_cmd","name":"save"}]}})";
      if (polls == 27) return R"({"ok":true,"result":{"events":[{"cmd":"prog_search","text":"plan","send":false}]}})";  // the as-you-type search must send it
      return R"({"ok":true,"result":{"events":[]}})";
    }
    if (tool == "prog.files") return R"({"ok":true,"result":{"root":"/work/prog","files":["plan.py","sub/n.toml"]}})";
    if (tool == "prog.read") return R"({"ok":true,"result":{"text":"def plan(x):\n    return x + 1\n"}})";
    if (tool == "prog.write") {
      written = j.at("args").s("text");
      writes.push_back(written);
      return R"({"ok":true,"result":{}})";
    }
    if (tool == "prog.search") return R"({"ok":true,"result":{"matches":[{"path":"plan.py","line":0,"col":4,"text":"def plan(x):"}],"truncated":false}})";
    if (tool == "prog.check") return R"({"ok":true,"result":{"problems":[]}})";
    if (tool == "term.open") return R"({"ok":true,"result":{"id":7,"cwd":"/work/prog"}})";
    if (tool == "term.read") return R"({"ok":true,"result":{"data":"\u001b[32mhi\u001b[0m\r\n","alive":true}})";
    if (tool == "library.categories") return R"({"ok":true,"result":{"categories":[]}})";
    if (tool == "library.search") return R"({"ok":true,"result":{"total":0,"parts":[]}})";
    if (tool == "library.ranges") return R"({"ok":true,"result":{"ranges":{}}})";
    return R"({"ok":true,"result":{},"state":{"rev":"r0","xml":")" + std::string(kModel) + R"(","build_error":"","path":"","undo":0,"redo":0,
      "report":{"assembly":"pendulum_demo","mass":{"total":1,"by_group":{}},"instances":[{"id":"arm","part":"demo-rod","group":"frame","body":"pole","mass":1}],
        "targets":[],"findings":[]},"ports":{"arm":[]}}})";
  };
  run_viewer(h, o);
  finish(h);
  const auto count = [&](const std::string& s) { return std::count(seen.begin(), seen.end(), s); };
  EXPECT_GE(count("prog.files"), 1);
  EXPECT_EQ(count("prog.read"), 1);
  ASSERT_EQ(writes.size(), 2u);
  EXPECT_EQ(writes[0], "# edited\ndef plan(x):\n    return x + 1\n") << "the typed text did not reach prog.write (escapes of newlines?)";
  EXPECT_EQ(writes[1], "# changed\ndef plan(x):\n    return x + 1\n") << "prog_find + prog_replace_all did not edit the buffer";
  EXPECT_GE(count("prog.search"), 1) << "find in files did not search as the query was typed";
  EXPECT_GE(count("prog.check"), 1);
  EXPECT_GE(count("term.read"), 2);   // the shell is polled while the page is open
  EXPECT_EQ(count("term.write"), 1);  // prog_term typed a line
  EXPECT_EQ(count("term.kill"), 1);   // the shell dies with the window
}

// ---- PROGRAM page Run slot: Run saves the dirty buffers, then run.start {mode:"bench"}, then run.poll until Stop sends run.stop ----
TEST(viewer_design, program_run_slot_saves_then_starts_polls_and_stops_a_bench_run) {
  NEED_DISPLAY();
  std::vector<std::string> seen;
  std::string start_args;
  int polls = 0, run_polls = 0;
  bool stopped = false;
  Host h(pendulum(), session());
  ViewerOptions o;
  o.frames = 1500;  // (same: ui_poll-timed events)
  o.width = 1100;
  o.height = 540;
  o.design_api = [&](const std::string& req) -> std::string {
    const design::Json j = design::Parser(req).parse();
    const std::string tool = j.s("tool");
    seen.push_back(tool);
    if (tool == "ui_poll") {
      ++polls;
      if (polls == 2) return R"({"ok":true,"result":{"events":[{"cmd":"mode","name":"program"}]}})";
      if (polls == 5) return R"({"ok":true,"result":{"events":[{"cmd":"prog_open","path":"plan.py","line":0,"col":0}]}})";
      if (polls == 9) return R"({"ok":true,"result":{"events":[{"cmd":"prog_type","text":"# edited\n"}]}})";
      if (polls == 12) return R"({"ok":true,"result":{"events":[{"cmd":"prog_cmd","name":"bench"},{"cmd":"prog_cmd","name":"run"}]}})";
      if (polls == 30) return R"({"ok":true,"result":{"events":[{"cmd":"prog_cmd","name":"stop"}]}})";
      return R"({"ok":true,"result":{"events":[]}})";
    }
    if (tool == "run.targets") return R"({"ok":true,"result":{"scenarios":["a.scenario.toml"],"components":[{"id":"arm","part":"demo-rod","kind":"servo"}]}})";
    if (tool == "run.start") {
      start_args = req;
      return R"({"ok":true,"result":{"run_id":"r1"}})";
    }
    if (tool == "run.poll") {
      ++run_polls;
      return stopped ? R"({"ok":true,"result":{"state":"stopped","out":"stopped\n","result":{"pass":false,"metrics":{}}}})"
                     : R"({"ok":true,"result":{"state":"running","out":"[bench] arm: servo\n"}})";
    }
    if (tool == "run.stop") {
      stopped = true;
      return R"({"ok":true,"result":{"state":"running"}})";
    }
    if (tool == "prog.files") return R"({"ok":true,"result":{"root":"/work/prog","files":["plan.py"]}})";
    if (tool == "prog.read") return R"({"ok":true,"result":{"text":"def plan(x):\n    return x + 1\n"}})";
    if (tool == "prog.write") return R"({"ok":true,"result":{}})";
    if (tool == "prog.check") return R"({"ok":true,"result":{"problems":[]}})";
    if (tool == "term.open") return R"({"ok":true,"result":{"id":7,"cwd":"/work/prog"}})";
    if (tool == "term.read") return R"({"ok":true,"result":{"data":"","alive":true}})";
    if (tool == "library.categories") return R"({"ok":true,"result":{"categories":[]}})";
    if (tool == "library.search") return R"({"ok":true,"result":{"total":0,"parts":[]}})";
    if (tool == "library.ranges") return R"({"ok":true,"result":{"ranges":{}}})";
    return R"({"ok":true,"result":{},"state":{"rev":"r0","xml":")" + std::string(kModel) + R"(","build_error":"","path":"","undo":0,"redo":0,
      "report":{"assembly":"pendulum_demo","mass":{"total":1,"by_group":{}},"instances":[{"id":"arm","part":"demo-rod","group":"frame","body":"pole","mass":1}],
        "targets":[],"findings":[]},"ports":{"arm":[]}}})";
  };
  run_viewer(h, o);
  finish(h);
  const auto first = [&](const std::string& s) { return std::find(seen.begin(), seen.end(), s) - seen.begin(); };
  const auto count = [&](const std::string& s) { return std::count(seen.begin(), seen.end(), s); };
  ASSERT_EQ(count("run.start"), 1);
  EXPECT_LT(first("prog.write"), first("run.start")) << "the dirty buffer must be saved before the run starts";
  EXPECT_LT(first("run.start"), first("run.poll"));
  EXPECT_NE(start_args.find("\"mode\":\"bench\""), std::string::npos) << start_args;
  EXPECT_NE(start_args.find("\"component\":\"arm\""), std::string::npos) << start_args;
  EXPECT_NE(start_args.find("\"files\":[\"plan.py\"]"), std::string::npos) << start_args;  // the open file, for its tests
  EXPECT_GE(run_polls, 3);          // polled repeatedly while running ...
  EXPECT_EQ(count("run.stop"), 1);  // ... and Stop sent run.stop
  EXPECT_TRUE(stopped);
}

// ---- a FULL SIM run leaves its final scene in the World: run.restore only when the window leaves for a design page ----
TEST(viewer_design, full_run_scene_is_held_until_the_window_leaves_for_a_design_page) {
  NEED_DISPLAY();
  std::vector<std::string> seen;
  int polls = 0, run_polls = 0, done_at = 0;
  Host h(pendulum(), session());
  ViewerOptions o;
  o.frames = 2000;  // events are timed by ui_poll count (100 ms wall clock each): frames must outlast them on a fast GPU
  o.width = 1100;
  o.height = 540;
  o.design_api = [&](const std::string& req) -> std::string {
    const std::string tool = design::Parser(req).parse().s("tool");
    seen.push_back(tool);
    if (tool == "ui_poll") {
      ++polls;
      if (polls == 2) return R"({"ok":true,"result":{"events":[{"cmd":"mode","name":"program"}]}})";  // (the target lists load when the page opens)
      if (polls == 12) return R"({"ok":true,"result":{"events":[{"cmd":"prog_cmd","name":"full"},{"cmd":"prog_cmd","name":"run"}]}})";
      if (done_at && polls == done_at + 10) return R"({"ok":true,"result":{"events":[{"cmd":"mode","name":"simulate"}]}})";  // the verdict banner is drawn here
      if (done_at && polls == done_at + 20) return R"({"ok":true,"result":{"events":[{"cmd":"mode","name":"program"}]}})";   // PROGRAM keeps the scene
      if (done_at && polls == done_at + 30) return R"({"ok":true,"result":{"events":[{"cmd":"mode","name":"build"}]}})";    // BUILD loads the design back
      return R"({"ok":true,"result":{"events":[]}})";
    }
    if (tool == "run.targets") return R"({"ok":true,"result":{"scenarios":["a.scenario.toml"],"components":[]}})";
    if (tool == "run.start") return R"({"ok":true,"result":{"run_id":"r1"}})";
    if (tool == "run.poll") {
      if (++run_polls == 3) done_at = polls;
      return run_polls < 3 ? R"({"ok":true,"result":{"state":"running","out":""}})"
                             : R"({"ok":true,"result":{"state":"done","out":"","result":{"pass":false,"metrics":{},"monitors":[{"name":"stays_up","pass":true},{"name":"in_cage","pass":false}],"fail_reasons":["in_cage: left the cage"]}}})";
    }
    if (tool == "run.restore") return R"({"ok":true,"result":{"restored":true}})";
    if (tool == "prog.files") return R"({"ok":true,"result":{"root":"/work/prog","files":[]}})";
    if (tool == "term.open") return R"({"ok":true,"result":{"id":7,"cwd":"/work/prog"}})";
    if (tool == "term.read") return R"({"ok":true,"result":{"data":"","alive":true}})";
    if (tool == "library.categories") return R"({"ok":true,"result":{"categories":[]}})";
    if (tool == "library.search") return R"({"ok":true,"result":{"total":0,"parts":[]}})";
    if (tool == "library.ranges") return R"({"ok":true,"result":{"ranges":{}}})";
    return R"({"ok":true,"result":{},"state":{"rev":"r0","xml":")" + std::string(kModel) + R"(","build_error":"","path":"","undo":0,"redo":0,
      "report":{"assembly":"pendulum_demo","mass":{"total":1,"by_group":{}},"instances":[{"id":"arm","part":"demo-rod","group":"frame","body":"pole","mass":1}],
        "targets":[],"findings":[]},"ports":{"arm":[]}}})";
  };
  run_viewer(h, o);
  finish(h);
  const auto first = [&](const std::string& s) { return std::find(seen.begin(), seen.end(), s) - seen.begin(); };
  ASSERT_EQ(std::count(seen.begin(), seen.end(), "run.start"), 1);
  ASSERT_EQ(std::count(seen.begin(), seen.end(), "run.restore"), 1) << "exactly one restore: the PROGRAM page keeps the scene, BUILD loads the design back";
  EXPECT_LT(first("run.poll"), first("run.restore"));
}

TEST(viewer_design, request_json_carries_file_text_and_keystrokes) {  // no window needed
  EXPECT_EQ(design::quote("a\nb\t\"c\"\x03"), R"("a\nb\t\"c\"\u0003")");
  const design::Json j = design::Parser(R"({"s":"x\u001b[0m 😀 é"})").parse();
  EXPECT_EQ(j.s("s"), "x\x1b[0m \xF0\x9F\x98\x80 \xC3\xA9");  // a surrogate pair is one 4-byte code point
}

// ---- camera.orbit / camera.fly: the backend's "camera_move" events glide the camera (no bus command; wall-clock time) -----------
TEST(viewer_design, camera_move_events_sweep_the_orbit_and_fly_the_eye) {
  NEED_DISPLAY();
  struct Cam {
    double pos[3], look[3];
  };
  std::vector<Cam> cams;
  std::atomic<bool> stop{false};
  std::atomic<std::size_t> nframes{0}, orbit_at{0}, fly_at{0};
  std::atomic<int> polls{0};
  Host h(pendulum(), session());
  ViewerOptions o;
  o.frames = 6000;  // (a safety net: the overlay stops the run as soon as the fly has arrived)
  o.width = 640;
  o.height = 360;
  o.clean = true;  // the 3D view fills the window: no panel shifts the camera
  o.stop = &stop;
  o.overlay = [&](const OverlayCtx& c) {
    cams.push_back({{c.cam_pos[0], c.cam_pos[1], c.cam_pos[2]}, {c.cam_look[0], c.cam_look[1], c.cam_look[2]}});
    nframes = cams.size();
    // the fly's end point is reached: eye (1, 2, 3), look (0, 0, 0.5)
    if (std::abs(c.cam_pos[0] - 1) < 1e-6 && std::abs(c.cam_pos[1] - 2) < 1e-6 && std::abs(c.cam_pos[2] - 3) < 1e-6) stop = true;
  };
  o.design_api = [&](const std::string& req) -> std::string {
    const design::Json j = design::Parser(req).parse();
    const std::string tool = j.s("tool");
    if (tool == "ui_poll") {
      const int n = ++polls;
      if (n == 6) {  // a full turn counter-clockwise over 0.8 s (a relative sweep: no shortest-way wrap)
        orbit_at = nframes.load();
        return R"({"ok":true,"result":{"events":[{"cmd":"camera_move","kind":"orbit","seconds":0.8,"ease":"linear","relative":true,"yaw_deg":360,"pitch_deg":0}]}})";
      }
      if (n == 30) {
        fly_at = nframes.load();
        return R"({"ok":true,"result":{"events":[{"cmd":"camera_move","kind":"fly","seconds":0.3,"ease":"smooth","eye":[1,2,3],"look":[0,0,0.5]}]}})";
      }
      return R"({"ok":true,"result":{"events":[]}})";
    }
    if (tool == "library.categories") return R"({"ok":true,"result":{"categories":[]}})";
    if (tool == "library.search") return R"({"ok":true,"result":{"total":0,"parts":[]}})";
    if (tool == "library.ranges") return R"({"ok":true,"result":{"ranges":{}}})";
    return R"({"ok":true,"result":{},"state":{"rev":"r0","xml":")" + std::string(kModel) + R"(","build_error":"","path":"","undo":0,"redo":0,
      "report":{"assembly":"pendulum_demo","mass":{"total":1,"by_group":{}},"instances":[{"id":"arm","part":"demo-rod","group":"frame","body":"pole","mass":1}],
        "targets":[],"findings":[]},"ports":{"arm":[]}}})";
  };
  run_viewer(h, o);
  h.stop();
  ASSERT_GT(orbit_at.load(), 0u);
  ASSERT_GT(fly_at.load(), orbit_at.load() + 5);
  const Cam ref = cams[orbit_at.load() - 1];
  const double dx0 = ref.pos[0] - ref.look[0], dy0 = ref.pos[1] - ref.look[1], dz0 = ref.pos[2] - ref.look[2], r0 = std::hypot(dx0, dy0);
  // the orbit: the eye keeps its radius and height; the azimuth sweeps through the far side (a shortest-way wrap would not move
  // at all for 360 degrees)
  double max_sep = 0;
  for (std::size_t i = orbit_at.load(); i < fly_at.load(); ++i) {
    const double dx = cams[i].pos[0] - cams[i].look[0], dy = cams[i].pos[1] - cams[i].look[1];
    EXPECT_NEAR(std::hypot(dx, dy), r0, 1e-6 * std::max(1.0, r0)) << "frame " << i << ": the orbit radius must stay";
    EXPECT_NEAR(cams[i].pos[2] - cams[i].look[2], dz0, 1e-6 * std::max(1.0, std::abs(dz0))) << "frame " << i << ": pitch must stay";
    max_sep = std::max(max_sep, std::hypot(dx - dx0, dy - dy0));
  }
  EXPECT_GT(max_sep, 1.3 * r0) << "the eye never swung to the far side";
  // the fly: ends exactly at the requested eye / look
  ASSERT_TRUE(stop.load()) << "the fly never arrived";
  EXPECT_NEAR(cams.back().look[0], 0, 1e-6);
  EXPECT_NEAR(cams.back().look[2], 0.5, 1e-6);
}
