#include "viewer/run_ui.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>

#include "viewer/theme.h"

namespace runui {
namespace {
std::string base(std::string s) {
  if (const auto k = s.rfind('/'); k != std::string::npos) s = s.substr(k + 1);
  if (s.ends_with(".scenario.toml")) s.resize(s.size() - 14);
  return s;
}
}  // namespace

std::string RunUi::chip() const { return running() ? "running" : verdict_; }

void RunUi::note(const std::string& text, const char* colour) {
  prog_.output_append(std::string(colour) + text + (*colour ? "\x1b[0m" : "") + "\r\n");
}

void RunUi::toggle() {
  if (running()) stop();
  else start();
}

bool RunUi::command(const std::string& name) {
  if (name == "bench" || name == "full") full_ = name == "full";
  else if (name == "run") start();
  else if (name == "stop") stop();
  else if (name.rfind("hover:", 0) == 0) hover_ = name.substr(6), click_ = 0;
  else if (name.rfind("click:", 0) == 0) hover_ = name.substr(6), click_ = 1;
  else if (name == "follow" || name == "unfollow") follow = name == "follow";  // the chase camera, as the user's take-over / resume
  else if (name.rfind("target:", 0) == 0) (full_ ? full_pick_ : bench_pick_) = name.substr(7);
  else return false;
  return true;
}

void RunUi::start() {
  if (running()) return;
  std::string args;
  if (full_) {
    const std::string sc = !full_pick_.empty() ? full_pick_ : scenarios_.empty() ? "" : scenarios_.front();
    if (sc.empty()) return;
    target_label_ = base(sc);
    args = "{\"mode\":\"full\",\"scenario\":" + design::quote(sc) + "}";
  } else {
    std::string id = bench_pick_;
    if (id.empty() && selected) id = selected();
    const auto has = [&](const std::string& i) { return std::any_of(comps_.begin(), comps_.end(), [&](const Comp& c) { return c.id == i; }); };
    if (!has(id)) id = comps_.empty() ? "" : comps_.front().id;
    if (id.empty()) return;
    target_label_ = id;
    const std::string f = prog_.active_path();
    args = "{\"mode\":\"bench\",\"component\":" + design::quote(id) + (f.empty() || f[0] == '@' || f.rfind("ext:", 0) == 0 ? "" : ",\"files\":[" + design::quote(f) + "]") + "}";
  }
  prog_.save_all();  // queued before run.start: the run sees the editor's text
  prog_.output_clear();
  prog_.show_output();
  state_ = State::Starting;
  verdict_.clear();
  held_ = false;  // the new run replaces the held scene (the Backend loads its own model)
  follow = true;
  stop_sent_ = false;
  send("run.start", design::request("run.start", args));
}

void RunUi::stop() {
  if (!running() || stop_sent_) return;
  if (state_ == State::Starting) return;  // the run id is not known yet; the button works again in a moment
  stop_sent_ = true;
  send("run.stop", design::request("run.stop", "{\"run_id\":" + design::quote(run_id_) + "}"));
}

bool RunUi::poll(double now, bool page_visible, std::string* kind, std::string* json) {
  if (page_visible && !was_visible_) targets_wanted_ = true;  // (re)read the component / scenario lists when the page opens
  was_visible_ = page_visible;
  if (targets_wanted_ && page_visible && !targets_flight_) {
    targets_wanted_ = false, targets_flight_ = true;
    *kind = "run.targets", *json = design::request("run.targets");
    return true;
  }
  if (state_ == State::Running && !poll_flight_ && now - last_poll_ >= 0.2) {
    last_poll_ = now, poll_flight_ = true;
    *kind = "run.poll", *json = design::request("run.poll", "{\"run_id\":" + design::quote(run_id_) + "}");
    return true;
  }
  return false;
}

void RunUi::on_reply(const std::string& kind, const design::Json& r) {
  const bool ok = r.at("ok").num != 0;
  const design::Json& res = r.at("result");
  if (kind == "run.targets") {
    targets_flight_ = false;
    if (!ok) return;
    scenarios_.clear(), comps_.clear();
    for (const design::Json& s : res.at("scenarios").arr) scenarios_.push_back(s.str);
    for (const design::Json& c : res.at("components").arr) comps_.push_back({c.s("id"), c.s("part"), c.s("kind")});
  } else if (kind == "run.start") {
    if (!ok) {
      note("run: " + r.s("error"), "\x1b[31m");
      state_ = State::Done, verdict_ = "failed";
      return;
    }
    run_id_ = res.s("run_id");
    state_ = State::Running;
    last_poll_ = 0;
  } else if (kind == "run.poll") {
    poll_flight_ = false;
    if (state_ != State::Running) return;
    if (!ok) {
      note("run: " + r.s("error"), "\x1b[31m");
      state_ = State::Done, verdict_ = "failed";
      return;
    }
    if (const std::string out = res.s("out"); !out.empty()) {
      std::string crlf;  // the OUTPUT view is a terminal: a bare \n would not return the carriage
      for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i] == '\n' && (i == 0 || out[i - 1] != '\r')) crlf += '\r';
        crlf += out[i];
      }
      prog_.output_append(crlf);
    }
    if (res.s("state") != "running") finish(res);
  } else if (kind == "run.stop" && !ok) {
    note("stop: " + r.s("error"), "\x1b[31m");
  }
}

void RunUi::finish(const design::Json& r) {
  const std::string st = r.s("state");
  const design::Json& result = r.at("result");
  const bool pass = result.at("pass").kind == design::Json::Bool || result.at("pass").kind == design::Json::Num ? result.at("pass").num != 0 : false;
  state_ = State::Done;
  verdict_ = st == "stopped" ? "stopped" : st == "failed" ? "failed" : pass ? "PASS" : "FAIL";
  monitors_.clear(), why_.clear();
  held_ = full_;  // (the Backend keeps the final scene of every full run, stopped ones too)
  if (full_) {
    for (const design::Json& m : result.at("monitors").arr) monitors_.push_back({m.s("name"), m.at("pass").num != 0});
    for (const design::Json& w : result.at("fail_reasons").arr) why_.push_back(w.str);
  }
  std::string line = "== " + verdict_ + "  " + (full_ ? "full: " : "bench: ") + target_label_;
  int n = 0;
  for (const auto& [k, v] : result.at("metrics").obj) {
    if (v.kind != design::Json::Num || n++ >= 4) continue;
    char b[64];
    std::snprintf(b, sizeof b, "  %s=%.4g", k.c_str(), v.num);
    line += b;
  }
  if (!result.s("run_dir").empty()) line += "  run dir " + result.s("run_dir");
  note(line, verdict_ == "PASS" ? "\x1b[32m" : verdict_ == "stopped" ? "\x1b[33m" : "\x1b[31m");
}

void RunUi::restore() {
  if (!held_ || running()) return;
  held_ = false;
  send("run.restore", design::request("run.restore"));
}

void RunUi::draw_result() {
  if (!held()) return;
  ImGui::Spacing();
  theme::caption("LAST FULL RUN");
  const ImVec4 col = verdict_ == "PASS" ? theme::kOk : verdict_ == "stopped" ? theme::kWarn : theme::kErr;
  ImGui::TextColored(col, "%s", verdict_.c_str());
  ImGui::SameLine();
  ImGui::TextWrapped("%s", target_label_.c_str());
  for (const auto& [name, ok] : monitors_) {
    ImGui::TextColored(ok ? theme::kOk : theme::kErr, "%s", ok ? "ok  " : "FAIL");
    ImGui::SameLine();
    ImGui::TextWrapped("%s", name.c_str());
  }
  for (const std::string& w : why_) ImGui::TextColored(theme::kErr, "why: %s", w.c_str());
  ImGui::TextDisabled("the scene above is the run's final state");
  if (theme::button("BACK TO PROGRAM", true)) back_to_program = true;
  ImGui::SetItemTooltip("the PROGRAM page; BUILD / LIBRARY / WIRING load the design's own model back");
}

void RunUi::draw(ImVec2 /*size*/) {
  const bool busy = running();
  // A disabled widget has no hover unless asked: say why instead of being silent.
  const auto busy_tip = [&](const char* what) {
    if (busy && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("a run is in progress: Stop it first\n(%s)", what);
  };
  const auto at = [&](const char* name) {  // scripted hover ("hover:<bench|full|combo|run>"): park the mouse on that control
    if (hover_ != name) return;
    const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent((a.x + b.x) / 2, (a.y + b.y) / 2);
    if (click_ && !io.MouseDown[0] && click_++ == 1) io.AddMouseButtonEvent(0, true);  // "click:<name>": press, then release a frame later
    else if (click_ && io.MouseDown[0]) io.AddMouseButtonEvent(0, false), click_ = 0;
  };
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {4, 0});
  ImGui::BeginDisabled(busy);
  if (theme::button("BENCH", !full_, {52, 0})) full_ = false;
  at("bench");
  if (busy) busy_tip("BENCH: the selected component's device model, then the tests of the open file");
  else ImGui::SetItemTooltip("the selected component's device model on a short bench, then the tests of the open file");
  ImGui::SameLine();
  if (theme::button("FULL SIM", full_, {66, 0})) full_ = true;
  at("full");
  if (busy) busy_tip("FULL SIM: the scenario's program, live in the 3D view");
  else ImGui::SetItemTooltip("the scenario's program, live in this window's 3D view");
  ImGui::EndDisabled();
  ImGui::SameLine();
  // target combo
  std::string cur, preview;  // cur: the target id (component id / scenario path); preview: what the closed combo shows
  const auto comp_label = [](const Comp& c) { return c.id + "   " + (c.kind.empty() ? "(tests only)" : c.kind) + (c.part.empty() ? "" : "   " + c.part); };
  if (full_) {
    cur = !full_pick_.empty() ? full_pick_ : scenarios_.empty() ? "" : scenarios_.front();
    preview = cur;
  } else {
    cur = bench_pick_;
    if (cur.empty() && selected) cur = selected();
    if (std::none_of(comps_.begin(), comps_.end(), [&](const Comp& c) { return c.id == cur; })) cur = comps_.empty() ? "" : comps_.front().id;
    for (const Comp& c : comps_)
      if (c.id == cur) preview = comp_label(c);
  }
  ImGui::BeginDisabled(busy);
  ImGui::SetNextItemWidth(140);  // (the run slot is 380 px wide: the open list is wider, and the tooltip names the full pick)
  ImGui::SetNextWindowSizeConstraints({420, 0}, {FLT_MAX, 340});
  if (ImGui::BeginCombo("##runtarget", preview.empty() ? "(none)" : preview.c_str())) {
    if (full_) {  // grouped by folder (relative to the program root), each entry shows its path
      std::string dir = "\x01";
      for (const std::string& s : scenarios_) {
        const std::string d = s.rfind('/') == std::string::npos ? "program root" : s.substr(0, s.rfind('/'));
        if (d != dir) ImGui::SeparatorText((dir = d).c_str());
        if (ImGui::Selectable(s.c_str(), s == cur)) full_pick_ = s;
      }
    } else {  // modelled components (the bench runs their device model) first, then the ones with tests only
      for (const bool modelled : {true, false}) {
        bool head = false;
        for (const Comp& c : comps_) {
          if (c.kind.empty() == modelled) continue;
          if (!head) ImGui::SeparatorText(modelled ? "DEVICE MODEL + TESTS" : "TESTS ONLY (no device model)"), head = true;
          if (ImGui::Selectable(comp_label(c).c_str(), c.id == cur)) bench_pick_ = c.id;
        }
      }
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  at("combo");
  if (busy) busy_tip(full_ ? "scenario to run" : "component to bench");
  else ImGui::SetItemTooltip("%s\n%s", preview.c_str(), full_ ? "scenario to run (*.scenario.toml in the program root)" : "component to bench (defaults to the one selected in BUILD)");
  ImGui::SameLine();
  const bool can = busy || !cur.empty();
  ImGui::BeginDisabled(!can || (busy && state_ == State::Starting));
  if (theme::button(busy ? "Stop" : "Run", !busy, {50, 0})) toggle();
  ImGui::EndDisabled();
  at("run");
  if (busy && state_ == State::Starting && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("the run is starting: Stop works in a moment");
  else ImGui::SetItemTooltip(busy ? "stop the run (kills the program)" : "save all, then run");
  if (const std::string c = chip(); !c.empty()) {
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(c == "PASS" ? theme::kOk : c == "running" ? theme::kAccent : c == "stopped" ? theme::kWarn : theme::kErr, "%s", c.c_str());
  }
  ImGui::PopStyleVar();
}

}  // namespace runui
