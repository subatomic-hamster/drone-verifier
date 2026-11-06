#pragma once
// The PROGRAM page: a small IDE over the design's program directory. Left: file explorer; centre: tabbed code editors
// (ImGuiColorTextEdit: syntax colouring, find / replace, multi-cursor, bracket matching, undo); bottom: TERMINAL (a pty line
// shell, ANSI colours), PROBLEMS, SEARCH (find in files), OUTPUT (for the RUN controls) and the session LOG. Python owns the
// files and the shells (python/dvsim/design/progops.py: prog.* / term.* ops); this side only draws and sends requests through
// the viewer's design queue (`send`), whose replies come back through `on_reply`.
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>

#include "design.h"

namespace progui {

struct State;  // everything below, defined in program_ui.cpp

class ProgramUi {
 public:
  ProgramUi();
  ~ProgramUi();

  // ---- wiring (set by the viewer) -------------------------------------------------------------------------------------
  // Queue one design-API request. `kind` ("prog.read:a.py", "term.read:2" ...) comes back to on_reply with the reply.
  std::function<void(const std::string& kind, const std::string& json)> send;

  // ---- the RUN toolbar slot -------------------------------------------------------------------------------------------
  // The strip at the top-right of the editor (kRunSlotW px wide, one toolbar row high). The RUN worker sets `run_slot`: it is
  // called every frame inside that strip with its size, cursor at the strip's top-left, mono font off (draw the Run / Test
  // button and the bench / full-simulation toggle there). Unset = a disabled "Run" stub.
  static constexpr float kRunSlotW = 380.f;
  std::function<void(ImVec2 size)> run_slot;

  // ---- called by the viewer -------------------------------------------------------------------------------------------
  // Draw the page into the current window (the viewer's PROGRAM pane). `log` = the session log (LOG tab).
  void draw(const std::deque<std::string>& log, bool has_backend);
  // A reply to a request this object sent (kind "prog.*" / "term.*"; `reply` = the whole {"ok":..,"result":..} object).
  void on_reply(const std::string& kind, const design::Json& reply);
  // Idle-time background request (terminal reads, first file listing): true + kind/json when one is due. Cheap.
  bool poll(double now, bool page_visible, std::string* kind, std::string* json);
  // ui.program: a source file shown instead of the editors' files (name empty = back to the design TOML).
  void show_external(const std::string& name, const std::string& text, float scroll);
  // Ribbon / script commands: save, save_all, find, replace, goto_line, quick_open, find_in_files, new_terminal, clear_terminal,
  // close_tab, reopen_tab, toggle_comment, design.
  void command(const std::string& cmd);
  void shutdown();  // kill the shells (the viewer is closing)
  // Scripted UI (backend event {"cmd":"prog_*", ...}, see progops.py `prog.ui`): prog_open {path,line?,col?,len?} (len selects),
  // prog_type {text} (at the cursor), prog_cmd {name} (= command), prog_term {text} (typed + Enter in the current terminal),
  // prog_hover {line,col} (holds the mouse over that glyph; line -1 releases), prog_dock {name}.
  void script(const design::Json& event);

  // ---- for the RUN worker ---------------------------------------------------------------------------------------------
  void save_all();                             // prog.write of every dirty buffer (queued before whatever is sent next)
  std::string active_path() const;             // relative path of the focused editor tab ("" = none)
  void output_append(const std::string& text);  // OUTPUT tab (ANSI colours understood)
  void output_clear();
  void show_output();                          // switch the bottom dock to OUTPUT
  void open_file(const std::string& path, int line = -1, int col = -1);

  // introspection for tests
  int open_buffers() const;
  bool any_dirty() const;
  int terminal_count() const;

 private:
  std::unique_ptr<State> s_;
};

}  // namespace progui
