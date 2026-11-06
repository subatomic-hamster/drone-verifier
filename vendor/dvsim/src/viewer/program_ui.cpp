#include "viewer/program_ui.h"

#include <TextEditor.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <set>

#include "viewer/theme.h"

namespace progui {
namespace {

using design::Json;
using theme::u32;

// ---- small helpers ----------------------------------------------------------------------------------------------------------
std::string ext_of(const std::string& p) {
  const auto d = p.rfind('.');
  const auto s = p.rfind('/');
  if (d == std::string::npos || (s != std::string::npos && d < s)) return "";
  std::string e = p.substr(d);
  for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e;
}
std::string base_of(const std::string& p) { return p.substr(p.rfind('/') == std::string::npos ? 0 : p.rfind('/') + 1); }
void put_utf8(std::string& o, char32_t c) {
  if (c < 0x80) o += static_cast<char>(c);
  else if (c < 0x800) o += static_cast<char>(0xC0 | (c >> 6)), o += static_cast<char>(0x80 | (c & 0x3F));
  else if (c < 0x10000)
    o += static_cast<char>(0xE0 | (c >> 12)), o += static_cast<char>(0x80 | ((c >> 6) & 0x3F)), o += static_cast<char>(0x80 | (c & 0x3F));
  else
    o += static_cast<char>(0xF0 | (c >> 18)), o += static_cast<char>(0x80 | ((c >> 12) & 0x3F)),
        o += static_cast<char>(0x80 | ((c >> 6) & 0x3F)), o += static_cast<char>(0x80 | (c & 0x3F));
}
std::string tag_of(const std::string& kind) {
  const auto c = kind.find(':');
  return c == std::string::npos ? "" : kind.substr(c + 1);
}
std::string tool_of(const std::string& kind) { return kind.substr(0, kind.find(':')); }
std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}
// subsequence match score for quick-open: lower is better, -1 = no match
int fuzzy(const std::string& hay, const std::string& needle) {
  if (needle.empty()) return 0;
  const std::string h = lower(hay), n = lower(needle);
  std::size_t j = 0;
  int gaps = 0;
  for (std::size_t i = 0; i < h.size() && j < n.size(); ++i)
    if (h[i] == n[j]) ++j;
    else if (j > 0) ++gaps;
  if (j < n.size()) return -1;
  return gaps + static_cast<int>(h.size()) / 4 - (lower(base_of(hay)).find(n) != std::string::npos ? 100 : 0);
}

// ---- languages -------------------------------------------------------------------------------------------------------------
const TextEditor::Language* toml_language() {
  static TextEditor::Language l = [] {
    TextEditor::Language t = *TextEditor::Language::Python();  // same comments, quotes, triple-quoted strings, numbers
    t.name = "TOML";
    t.keywords = {"true", "false"};
    t.indentationForBlocks = false;
    return t;
  }();
  return &l;
}
const TextEditor::Language* lang_for(const std::string& path) {
  const std::string e = ext_of(path);
  if (e == ".py" || e == ".pyi") return TextEditor::Language::Python();
  if (e == ".toml") return toml_language();
  if (e == ".json") return TextEditor::Language::Json();
  if (e == ".c") return TextEditor::Language::C();
  if (e == ".h" || e == ".hpp" || e == ".hh" || e == ".cpp" || e == ".cc" || e == ".cxx" || e == ".ino" || e == ".cu") return TextEditor::Language::Cpp();
  if (e == ".md") return TextEditor::Language::Markdown();
  if (e == ".glsl" || e == ".vert" || e == ".frag") return TextEditor::Language::Glsl();
  return nullptr;
}

// the editor's colours: VS Code Dark+ tokens on the viewer's own greys
const TextEditor::Palette& palette() {
  static const TextEditor::Palette p = {{
      IM_COL32(220, 222, 228, 255),  // text
      IM_COL32(197, 134, 192, 255),  // keyword
      IM_COL32(86, 156, 214, 255),   // declaration
      IM_COL32(181, 206, 168, 255),  // number
      IM_COL32(206, 145, 120, 255),  // string
      IM_COL32(170, 176, 190, 255),  // punctuation
      IM_COL32(155, 155, 155, 255),  // preprocessor
      IM_COL32(156, 220, 254, 255),  // identifier
      IM_COL32(78, 201, 176, 255),   // known identifier
      IM_COL32(106, 153, 85, 255),   // comment
      IM_COL32(27, 28, 31, 255),     // background (a little below kBg)
      IM_COL32(240, 240, 245, 255),  // cursor
      IM_COL32(40, 90, 160, 255),    // selection
      IM_COL32(70, 72, 80, 255),     // whitespace
      IM_COL32(60, 64, 74, 255),     // matching bracket background
      IM_COL32(130, 138, 156, 255),  // matching bracket active
      IM_COL32(246, 222, 36, 255),   // levels
      IM_COL32(66, 140, 230, 255),
      IM_COL32(213, 96, 213, 255),
      IM_COL32(220, 50, 60, 255),    // bracket error
      IM_COL32(110, 115, 128, 255),  // line number
      IM_COL32(220, 224, 236, 255),  // current line number
  }};
  return p;
}

// ---- terminal ----------------------------------------------------------------------------------------------------------------
constexpr std::uint32_t kTrue = 1u << 31, kBold = 1u << 30, kDimA = 1u << 29, kDefault = 0x1FF;
const ImU32 kAnsi16[16] = {IM_COL32(88, 92, 102, 255),   IM_COL32(205, 49, 49, 255),   IM_COL32(13, 188, 121, 255),  IM_COL32(229, 229, 16, 255),
                           IM_COL32(36, 114, 200, 255),  IM_COL32(188, 63, 188, 255),  IM_COL32(17, 168, 205, 255),  IM_COL32(229, 229, 229, 255),
                           IM_COL32(120, 124, 134, 255), IM_COL32(241, 76, 76, 255),   IM_COL32(35, 209, 139, 255),  IM_COL32(245, 245, 67, 255),
                           IM_COL32(59, 142, 234, 255),  IM_COL32(214, 112, 214, 255), IM_COL32(41, 184, 219, 255),  IM_COL32(255, 255, 255, 255)};

ImU32 ansi_color(std::uint32_t at) {
  if (at & kTrue) return IM_COL32((at >> 16) & 255, (at >> 8) & 255, at & 255, 255);
  const unsigned i = at & 0x1FF;
  if (i == kDefault) return u32(theme::kText, (at & kDimA) ? 0.6f : 1.f);
  if (i < 16) return kAnsi16[(at & kBold) && i < 8 ? i + 8 : i];
  if (i < 232) {
    const unsigned k = i - 16, lv[6] = {0, 95, 135, 175, 215, 255};
    return IM_COL32(lv[k / 36], lv[(k / 6) % 6], lv[k % 6], 255);
  }
  const unsigned g = 8 + (i - 232) * 10;
  return IM_COL32(g, g, g, 255);
}

struct TermLine {
  std::u32string ch;
  std::vector<std::uint32_t> at;
};

struct Term {
  int id = 0;           // pty id from term.open (0 = still opening); -1 = a local view (OUTPUT)
  std::string key;      // local handle of the open request
  std::string title;
  bool alive = true, exited_noted = false, closing = false;
  std::deque<TermLine> lines{TermLine{}};
  std::size_t cx = 0;
  std::uint32_t attr = kDefault;
  enum { Plain, Esc, Csi, Osc, OscEsc, Skip1 } st = Plain;
  std::string params;
  char32_t cp = 0;
  int need = 0;
  std::vector<std::string> history;
  int hist_pos = 0;
  char input[2048] = {0};
  bool input_sel = false, stick = true, new_data = false, focus_input = false;
  double last_poll = 0;
  std::size_t max_cols = 0;
  bool sel_on = false, dragging = false;
  int sr0 = 0, sc0 = 0, sr1 = 0, sc1 = 0;  // selection anchor / cursor (row, col)

  void put(char32_t c) {
    TermLine& l = lines.back();
    while (l.ch.size() < cx) l.ch.push_back(U' '), l.at.push_back(kDefault);
    if (cx < l.ch.size()) l.ch[cx] = c, l.at[cx] = attr;
    else l.ch.push_back(c), l.at.push_back(attr);
    ++cx;
    max_cols = std::max(max_cols, l.ch.size());
  }
  void newline() {
    lines.emplace_back();
    cx = 0;
    if (lines.size() > 6000) {
      lines.pop_front();
      sel_on = false;
    }
  }
  void sgr() {
    std::vector<int> p;
    std::string cur;
    for (char c : params + ";") {
      if (c == ';' || c == ':') p.push_back(cur.empty() ? 0 : std::atoi(cur.c_str())), cur.clear();
      else cur += c;
    }
    for (std::size_t i = 0; i < p.size(); ++i) {
      const int n = p[i];
      if (n == 0) attr = kDefault;
      else if (n == 1) attr |= kBold;
      else if (n == 2) attr |= kDimA;
      else if (n == 22) attr &= ~(kBold | kDimA);
      else if (n == 39) attr = (attr & (kBold | kDimA)) | kDefault;
      else if (n >= 30 && n <= 37) attr = (attr & (kBold | kDimA)) | static_cast<unsigned>(n - 30);
      else if (n >= 90 && n <= 97) attr = (attr & (kBold | kDimA)) | static_cast<unsigned>(n - 90 + 8);
      else if ((n == 38) && i + 1 < p.size()) {
        const std::uint32_t keep = attr & (kBold | kDimA);
        if (p[i + 1] == 5 && i + 2 < p.size()) attr = keep | static_cast<unsigned>(p[i + 2] & 255), i += 2;
        else if (p[i + 1] == 2 && i + 4 < p.size())
          attr = keep | kTrue | (static_cast<unsigned>(p[i + 2] & 255) << 16) | (static_cast<unsigned>(p[i + 3] & 255) << 8) | static_cast<unsigned>(p[i + 4] & 255),
          i += 4;
      } else if (n == 48 && i + 1 < p.size()) {
        i += p[i + 1] == 5 ? 2 : 4;  // background colours are not drawn
      }
    }
  }
  void csi(char f) {
    int n0 = 0, n1 = 0;
    std::string digits;
    for (char c : params)
      if (std::isdigit(static_cast<unsigned char>(c))) digits += c;
      else break;
    if (!digits.empty()) n0 = std::atoi(digits.c_str());
    const auto semi = params.find(';');
    if (semi != std::string::npos) n1 = std::atoi(params.c_str() + semi + 1);
    TermLine& l = lines.back();
    const std::size_t cnt = static_cast<std::size_t>(std::max(n0, 1));
    switch (f) {
      case 'm': sgr(); break;
      case 'K':
        if (n0 == 0) {
          if (cx < l.ch.size()) l.ch.resize(cx), l.at.resize(cx);
        } else {
          for (std::size_t i = 0; i < (n0 == 2 ? l.ch.size() : std::min(cx, l.ch.size())); ++i) l.ch[i] = U' ';
        }
        break;
      case 'J':
        if (n0 == 2 || n0 == 3) lines.assign(1, TermLine{}), cx = 0, sel_on = false;
        else if (n0 == 0 && cx < l.ch.size()) l.ch.resize(cx), l.at.resize(cx);
        break;
      case 'C': cx += cnt; break;
      case 'D': cx = cx > cnt ? cx - cnt : 0; break;
      case 'G': cx = static_cast<std::size_t>(std::max(n0, 1) - 1); break;
      case 'H':
      case 'f': cx = static_cast<std::size_t>(std::max(n1, 1) - 1); break;
      case 'P':
        if (cx < l.ch.size()) {
          const std::size_t e = std::min(l.ch.size(), cx + cnt);
          l.ch.erase(l.ch.begin() + static_cast<long>(cx), l.ch.begin() + static_cast<long>(e));
          l.at.erase(l.at.begin() + static_cast<long>(cx), l.at.begin() + static_cast<long>(e));
        }
        break;
      case 'X':
        for (std::size_t i = cx; i < std::min(l.ch.size(), cx + cnt); ++i) l.ch[i] = U' ';
        break;
      default: break;  // cursor up / down, modes, scroll regions ...: a line shell has no use for them
    }
  }
  void feed(const std::string& data) {
    for (const unsigned char ch : data) {
      switch (st) {
        case Plain:
          if (ch == 0x1b) st = Esc;
          else if (ch == '\n') newline();
          else if (ch == '\r') cx = 0;
          else if (ch == '\b') cx = cx > 0 ? cx - 1 : 0;
          else if (ch == '\t') {
            do put(U' ');
            while (cx % 8);
          } else if (ch < 0x20 || ch == 0x7f) {
          } else if (ch < 0x80) put(ch);
          else if (ch >= 0xC0) need = ch >= 0xF0 ? 3 : ch >= 0xE0 ? 2 : 1, cp = ch & (0x3Fu >> need);
          else if (need > 0 && --need >= 0) {
            cp = (cp << 6) | (ch & 0x3F);
            if (need == 0) put(cp);
          }
          break;
        case Esc:
          if (ch == '[') st = Csi, params.clear();
          else if (ch == ']') st = Osc;
          else if (ch == '(' || ch == ')' || ch == '#') st = Skip1;
          else st = Plain;
          break;
        case Csi:
          if (ch >= 0x40 && ch <= 0x7e) csi(static_cast<char>(ch)), st = Plain;
          else params += static_cast<char>(ch);
          break;
        case Osc:
          if (ch == 7) st = Plain;
          else if (ch == 0x1b) st = OscEsc;
          break;
        case OscEsc:
        case Skip1: st = Plain; break;
      }
    }
    new_data = true;
  }
  std::string selected_text() const {
    int r0 = sr0, c0 = sc0, r1 = sr1, c1 = sc1;
    if (r0 > r1 || (r0 == r1 && c0 > c1)) std::swap(r0, r1), std::swap(c0, c1);
    std::string o;
    for (int r = std::max(r0, 0); r <= r1 && r < static_cast<int>(lines.size()); ++r) {
      const std::u32string& s = lines[static_cast<std::size_t>(r)].ch;
      std::string row;
      for (std::size_t c = (r == r0 ? static_cast<std::size_t>(std::max(c0, 0)) : 0); c < (r == r1 ? std::min<std::size_t>(c1 + 1, s.size()) : s.size()); ++c)
        put_utf8(row, s[c]);
      while (!row.empty() && row.back() == ' ') row.pop_back();
      o += row;
      if (r < r1) o += '\n';
    }
    return o;
  }
};

// ---- editor buffers ----------------------------------------------------------------------------------------------------------
struct Problem {
  int line = 0, col = 0, end_col = 0;
  std::string msg;
};

struct Buffer {
  std::string path, label;
  TextEditor ed;
  int saved_undo = 0, save_idx = 0;
  bool loaded = false, ro = false, check_due = false, check_flight = false;
  int jump_line = -1, jump_col = 0;
  float ext_scroll = -1.f;
  int sel_len = 0;  // scripted prog_open: select this many glyphs at the jump target (after the jump)
  std::string error;
  std::vector<Problem> problems;
  bool dirty() const { return !ro && loaded && ed.GetUndoIndex() != static_cast<std::size_t>(saved_undo); }
};

struct Peer {
  std::string component, port, pin;
};
struct PinRow {
  std::string component, port, pin;
  std::vector<Peer> connects;
};
struct Hover {
  std::string kind, id, part, markdown;
  std::vector<PinRow> pins;
  bool empty() const { return kind.empty() && markdown.empty(); }
};

struct Node {
  std::string name, path;
  bool file = false;
  std::vector<Node> kids;
};

struct Match {
  std::string path, text;
  int line = 0, col = 0;
};

enum class Dock { Terminal, Problems, Search, Output, Log };

}  // namespace

struct State {
  ProgramUi* self = nullptr;
  // files
  std::string root;
  std::vector<std::string> files;
  Node tree;
  std::set<std::string> open_dirs;
  bool files_wanted = true, files_flight = false, files_err = false;
  std::string files_error;
  // buffers
  std::vector<std::unique_ptr<Buffer>> bufs;
  int active = -1, want_tab = -1;
  std::vector<std::string> closed;  // reopen stack
  int close_ask = -1;
  // layout
  float exp_w = 230, bot_h = 230;
  Dock dock = Dock::Terminal;
  // terminals
  std::vector<std::unique_ptr<Term>> terms;
  int term_cur = 0, term_serial = 0;
  bool term_started = false;
  std::unique_ptr<Term> output = std::make_unique<Term>();
  std::size_t poll_rr = 0;
  // popups
  enum class Pop { None, Quick, Goto, Rename, NewFile } pop = Pop::None;
  char pop_text[512] = {0};
  int pop_sel = 0;
  bool pop_open = false, discard_req = false;  // requests to open the popups (see popups())
  // search
  char q[256] = {0};
  bool q_case = false, q_word = false, q_regex = false, q_focus = false, q_flight = false;
  std::vector<Match> matches;
  bool truncated = false;
  std::string q_error, q_sent, q_seen;  // q_sent: the query the results belong to; q_seen/q_t: typing debounce
  double q_t = 0;
  // find / replace bar: docked above the editor (the widget's own floating one is never opened). Matches are per line.
  struct FM {
    int line, col, len;  // col / len in code points
  };
  bool f_open = false, f_repl = false, f_focus = false, f_case = false, f_word = false, f_re = false, f_goto = false, f_from_set = false;
  char f_q[256] = {0}, f_r[256] = {0};
  std::vector<FM> f_m;
  int f_cur = -1, f_from_line = 0, f_from_col = 0;
  std::string f_err, f_key;
  // hover
  std::string hover_key;
  Hover hover;
  bool hover_valid = false;
  // misc
  std::string note;
  bool note_bad = false;
  double note_t = -100;
  std::string rename_path, rename_new;
  int rename_line = 0, rename_col = 0;
  bool rename_pending = false;
  bool hover_on = false;  // scripted: hold the mouse over (hover_line, hover_col)
  int hover_line = 0, hover_col = 0;
  std::function<void(const std::string&, const std::string&)>& send_ref;
  explicit State(std::function<void(const std::string&, const std::string&)>& s) : send_ref(s) {}

  // ---- requests --------------------------------------------------------------------------------------------------------------
  void req(const std::string& kind, const std::string& tool, const std::string& args) {
    if (send_ref) send_ref(kind, design::request(tool, args));
  }
  static std::string q_(const std::string& s) { return design::quote(s); }
  void say(const std::string& s, bool bad = false) {
    note = s, note_bad = bad, note_t = ImGui::GetTime();
  }
  Buffer* find(const std::string& path) {
    for (auto& b : bufs)
      if (b->path == path) return b.get();
    return nullptr;
  }
  Buffer* cur() { return active >= 0 && active < static_cast<int>(bufs.size()) ? bufs[static_cast<std::size_t>(active)].get() : nullptr; }

  void configure(Buffer& b) {
    TextEditor& e = b.ed;
    e.SetPalette(palette());
    e.SetTabSize(4);
    e.SetShowWhitespacesEnabled(false);
    e.SetInsertSpacesOnTabs(true);
    e.SetShowLineNumbersEnabled(true);
    e.SetShowMatchingBrackets(true);
    e.SetCompletePairedGlyphs(true);
    e.SetAutoIndentEnabled(true);
    e.SetShowScrollbarMiniMapEnabled(true);
    e.SetLineSpacing(1.12f);
    e.SetMiddleMouseScrollMode();
    e.SetFindButtonLabel("Next");
    Buffer* bp = &b;
    e.SetChangeCallback([bp] { bp->check_due = true; }, 450);
    e.SetTextHoverCallback([this, bp](TextEditor::PopupData d) { hover_popup(*bp, d); });
  }

  Buffer* open(const std::string& path, int line = -1, int col = 0, int sel_len = 0) {
    Buffer* b = find(path);
    if (!b) {
      bufs.push_back(std::make_unique<Buffer>());
      b = bufs.back().get();
      b->path = path, b->label = base_of(path);
      configure(*b);
      b->ed.SetLanguage(lang_for(path));
      if (path == "@design") b->label = "design.toml", b->ro = true, b->ed.SetReadOnlyEnabled(true), req("prog.design", "design_text", "{}");
      else req("prog.read:" + path, "prog.read", "{\"path\":" + q_(path) + "}");
    }
    want_tab = static_cast<int>(std::find_if(bufs.begin(), bufs.end(), [&](auto& x) { return x.get() == b; }) - bufs.begin());
    active = want_tab;
    if (line >= 0) {
      b->jump_line = line, b->jump_col = col, b->sel_len = sel_len;
      if (b->loaded) jump(*b);
    }
    return b;
  }
  void jump(Buffer& b) {
    b.ed.SetCursor(TextEditor::DocPos(static_cast<std::size_t>(b.jump_line), static_cast<std::size_t>(std::max(b.jump_col, 0))));
    b.ed.ScrollToLine(static_cast<std::size_t>(b.jump_line), TextEditor::Scroll::alignMiddle);
    if (b.sel_len > 0)
      b.ed.SelectRegion(TextEditor::DocPos(static_cast<std::size_t>(b.jump_line), static_cast<std::size_t>(b.jump_col)),
                        TextEditor::DocPos(static_cast<std::size_t>(b.jump_line), static_cast<std::size_t>(b.jump_col + b.sel_len)));
    b.sel_len = 0;
    b.ed.SetFocus();
    b.jump_line = -1;
  }
  void save(Buffer& b) {
    if (b.ro || !b.loaded) return;
    b.save_idx = static_cast<int>(b.ed.GetUndoIndex());
    req("prog.write:" + b.path, "prog.write", "{\"path\":" + q_(b.path) + ",\"text\":" + q_(b.ed.GetText()) + "}");
  }
  void save_all() {
    for (auto& b : bufs)
      if (b->dirty()) save(*b);
  }
  void close_tab(int i, bool force) {
    if (i < 0 || i >= static_cast<int>(bufs.size())) return;
    Buffer& b = *bufs[static_cast<std::size_t>(i)];
    if (b.dirty() && !force) {
      close_ask = i;
      discard_req = true;
      return;
    }
    if (b.path.rfind("ext:", 0) != 0 && b.path != "@design") closed.push_back(b.path);
    bufs.erase(bufs.begin() + i);
    if (active >= static_cast<int>(bufs.size())) active = static_cast<int>(bufs.size()) - 1;
    want_tab = active;
  }
  void rebuild_tree() {
    tree = Node{};
    for (const std::string& f : files) {
      Node* n = &tree;
      std::size_t a = 0;
      std::string acc;
      for (;;) {
        const auto s = f.find('/', a);
        const std::string part = f.substr(a, s == std::string::npos ? std::string::npos : s - a);
        acc += (acc.empty() ? "" : "/") + part;
        const bool leaf = s == std::string::npos;
        auto it = std::find_if(n->kids.begin(), n->kids.end(), [&](const Node& k) { return k.name == part && k.file == leaf; });
        if (it == n->kids.end()) n->kids.push_back(Node{part, acc, leaf, {}}), it = n->kids.end() - 1;
        n = &*it;
        if (leaf) break;
        a = s + 1;
      }
    }
    const std::function<void(Node&)> sort = [&](Node& n) {
      std::stable_sort(n.kids.begin(), n.kids.end(), [](const Node& a, const Node& b) { return a.file != b.file ? !a.file : a.name < b.name; });
      for (Node& k : n.kids) sort(k);
    };
    sort(tree);
  }

  // ---- editor-side actions ---------------------------------------------------------------------------------------------------
  std::string word_at_cursor(Buffer& b) {
    const TextEditor::DocPos p = b.ed.GetMainCursorPosition();
    const std::string l = b.ed.GetLineText(p.line);
    std::size_t s = std::min(p.index, l.size()), e = s;
    const auto w = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || (static_cast<unsigned char>(c) & 0x80); };
    while (s > 0 && w(l[s - 1])) --s;
    while (e < l.size() && w(l[e])) ++e;
    return l.substr(s, e - s);
  }
  void do_define(Buffer& b) {
    const TextEditor::DocPos p = b.ed.GetMainCursorPosition();
    req("prog.define:" + b.path, "prog.define",
        "{\"path\":" + q_(b.path) + ",\"line\":" + std::to_string(p.line) + ",\"col\":" + std::to_string(p.index) + ",\"text\":" + q_(b.ed.GetText()) + "}");
  }
  void start_rename(Buffer& b) {
    if (ext_of(b.path) != ".py") return say("rename works on python files", true);
    std::snprintf(pop_text, sizeof pop_text, "%s", word_at_cursor(b).c_str());
    pop = Pop::Rename, pop_open = true;
  }
  void run_rename(Buffer& b, const std::string& nw) {
    const TextEditor::DocPos p = b.ed.GetMainCursorPosition();
    save_all();  // jedi reads the disk: the buffers go first (the queue keeps the order)
    rename_path = b.path, rename_line = static_cast<int>(p.line), rename_col = static_cast<int>(p.index), rename_new = nw;
    req("prog.rename", "prog.rename",
        "{\"path\":" + q_(b.path) + ",\"line\":" + std::to_string(rename_line) + ",\"col\":" + std::to_string(rename_col) + ",\"new\":" + q_(nw) + "}");
  }
  std::string q_key() const { return std::string(q) + (q_case ? "|c" : "|i") + (q_word ? "w" : "-") + (q_regex ? "r" : "-"); }

  // ---- find / replace in the open file --------------------------------------------------------------------------------------
  static bool word_ch(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || (static_cast<unsigned char>(c) & 0x80); }
  static int cps(const std::string& s, std::size_t from, std::size_t to) {  // code points in s[from, to)
    int n = 0;
    for (std::size_t i = from; i < to && i < s.size(); ++i) n += (static_cast<unsigned char>(s[i]) & 0xC0) != 0x80;
    return n;
  }
  void find_open(Buffer* b, bool repl) {
    f_open = true, f_focus = true;
    if (repl && b && !b->ro) f_repl = true;
    if (b && b->ed.CurrentCursorHasSelection()) {
      const std::string t = b->ed.GetCursorText(0);
      if (t.find('\n') == std::string::npos && t.size() < sizeof f_q) std::snprintf(f_q, sizeof f_q, "%s", t.c_str());
    } else if (b && !f_q[0]) {
      std::snprintf(f_q, sizeof f_q, "%s", word_at_cursor(*b).c_str());
    }
    f_key.clear(), f_goto = true;
  }
  void find_close(Buffer* b) {
    f_open = false, f_m.clear(), f_cur = -1;
    if (b) b->ed.SetFocus();
  }
  void goto_match(Buffer& b, int i) {
    if (i < 0 || i >= static_cast<int>(f_m.size())) return;
    const FM& m = f_m[static_cast<std::size_t>(i)];
    const TextEditor::DocPos a(static_cast<std::size_t>(m.line), static_cast<std::size_t>(m.col));
    b.ed.SelectRegion(a, TextEditor::DocPos(a.line, a.index + static_cast<std::size_t>(m.len)));
    if (!b.ed.IsDocPosVisible(a)) b.ed.ScrollToLine(a.line, TextEditor::Scroll::alignMiddle);
    f_cur = i;
  }
  // first match at or after (line, col); wraps; -1 when there are none
  int match_from(int line, int col) const {
    for (std::size_t i = 0; i < f_m.size(); ++i)
      if (f_m[i].line > line || (f_m[i].line == line && f_m[i].col >= col)) return static_cast<int>(i);
    return f_m.empty() ? -1 : 0;
  }
  void find_update(Buffer& b) {
    if (!f_open || !b.loaded) return;
    const std::string key = b.path + "|" + std::to_string(b.ed.GetUndoIndex()) + "|" + std::to_string(b.ed.GetLineCount()) + "|" + f_q + (f_case ? "|c" : "|i") +
                            (f_word ? "w" : "-") + (f_re ? "r" : "-");
    if (key == f_key) return;
    f_key = key;
    f_m.clear(), f_err.clear(), f_cur = -1;
    if (f_q[0]) {
      const std::string text = b.ed.GetText();
      std::regex rx;
      if (f_re) {
        try {
          rx = std::regex(f_q, f_case ? std::regex::ECMAScript : (std::regex::ECMAScript | std::regex::icase));
        } catch (const std::regex_error&) {
          f_err = "invalid regular expression";
        }
      }
      const std::string needle = f_case ? std::string(f_q) : lower(f_q);
      std::size_t pos = 0;
      int ln = 0;
      while (f_err.empty() && pos <= text.size() && f_m.size() < 5000) {
        std::size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        std::string line = text.substr(pos, nl - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto add = [&](std::size_t p, std::size_t n) {
          if (n == 0 || (f_word && ((p > 0 && word_ch(line[p - 1])) || (p + n < line.size() && word_ch(line[p + n]))))) return;
          f_m.push_back({ln, cps(line, 0, p), cps(line, p, p + n)});
        };
        if (f_re) {
          for (std::sregex_iterator it(line.begin(), line.end(), rx), end; it != end; ++it) add(static_cast<std::size_t>(it->position()), static_cast<std::size_t>(it->length()));
        } else {
          const std::string hay = f_case ? line : lower(line);
          for (std::size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + needle.size())) add(p, needle.size());
        }
        pos = nl + 1;
        ++ln;
      }
    }
    if (f_m.empty()) return;
    int l = 0, c = 0;
    if (f_from_set) l = f_from_line, c = f_from_col, f_from_set = false;
    else {
      const TextEditor::DocPos p = b.ed.GetMainCursorSelection().start;
      l = static_cast<int>(p.line), c = static_cast<int>(p.index);
    }
    f_cur = match_from(l, c);
    if (f_goto) goto_match(b, f_cur), f_goto = false;
  }
  void find_step(Buffer& b, int dir) {
    if (f_m.empty()) return;
    const int n = static_cast<int>(f_m.size());
    const TextEditor::DocSelection sel = b.ed.GetMainCursorSelection();
    const int at = match_from(static_cast<int>(sel.start.line), static_cast<int>(sel.start.index));
    const bool on = f_cur >= 0 && f_m[static_cast<std::size_t>(f_cur)].line == static_cast<int>(sel.start.line) &&
                    f_m[static_cast<std::size_t>(f_cur)].col == static_cast<int>(sel.start.index);
    int t = on ? f_cur + dir : (dir > 0 ? at : at - 1);
    goto_match(b, ((t % n) + n) % n);
  }
  std::string replacement(Buffer& b, const FM& m) {
    if (!f_re) return f_r;
    const TextEditor::DocPos a(static_cast<std::size_t>(m.line), static_cast<std::size_t>(m.col));
    const std::string hit = b.ed.GetSectionText(a, TextEditor::DocPos(a.line, a.index + static_cast<std::size_t>(m.len)));
    try {
      return std::regex_replace(hit, std::regex(f_q, f_case ? std::regex::ECMAScript : (std::regex::ECMAScript | std::regex::icase)), std::string(f_r),
                                std::regex_constants::format_first_only);
    } catch (const std::regex_error&) {
      return f_r;
    }
  }
  void replace_one(Buffer& b) {
    if (b.ro || f_m.empty()) return;
    int i = f_cur;
    const TextEditor::DocSelection sel = b.ed.GetMainCursorSelection();
    if (i < 0 || f_m[static_cast<std::size_t>(i)].line != static_cast<int>(sel.start.line) || f_m[static_cast<std::size_t>(i)].col != static_cast<int>(sel.start.index))
      return goto_match(b, match_from(static_cast<int>(sel.start.line), static_cast<int>(sel.start.index)));  // first press selects the match, the next replaces it
    const FM m = f_m[static_cast<std::size_t>(i)];
    const std::string rep = replacement(b, m);
    const TextEditor::DocPos a(static_cast<std::size_t>(m.line), static_cast<std::size_t>(m.col));
    b.ed.ReplaceSectionText(a, TextEditor::DocPos(a.line, a.index + static_cast<std::size_t>(m.len)), rep);
    f_from_set = true, f_from_line = m.line, f_from_col = m.col + cps(rep, 0, rep.size());
    f_goto = true, f_key.clear();
  }
  void replace_all(Buffer& b) {
    if (b.ro || f_m.empty()) return;
    const std::vector<FM> ms = f_m;
    std::vector<std::string> reps;
    for (const FM& m : ms) reps.push_back(replacement(b, m));
    for (std::size_t i = ms.size(); i-- > 0;) {  // bottom-up: earlier positions stay valid
      const TextEditor::DocPos a(static_cast<std::size_t>(ms[i].line), static_cast<std::size_t>(ms[i].col));
      b.ed.ReplaceSectionText(a, TextEditor::DocPos(a.line, a.index + static_cast<std::size_t>(ms[i].len)), reps[i]);
    }
    say("replaced " + std::to_string(ms.size()) + " occurrence(s)");
    f_key.clear();
  }

  void search_files() {
    if (!q[0]) return;
    q_flight = true;
    q_sent = q_key();
    req("prog.search", "prog.search",
        "{\"query\":" + q_(q) + ",\"case\":" + (q_case ? "true" : "false") + ",\"word\":" + (q_word ? "true" : "false") + ",\"regex\":" +
            (q_regex ? "true" : "false") + "}");
  }
  void focus_search(Buffer* b) {
    dock = Dock::Search, q_focus = true;
    if (b && b->ed.CurrentCursorHasSelection()) {
      const std::string t = b->ed.GetCursorText(0);
      if (t.find('\n') == std::string::npos && t.size() < sizeof q) std::snprintf(q, sizeof q, "%s", t.c_str());
    }
  }
  void new_term() {
    terms.push_back(std::make_unique<Term>());
    Term& t = *terms.back();
    t.key = std::to_string(++term_serial);
    t.title = "shell " + t.key;
    t.focus_input = true;
    term_cur = static_cast<int>(terms.size()) - 1;
    dock = Dock::Terminal;
    req("term.open:" + t.key, "term.open", "{}");
  }
  void kill_term(int i) {
    if (i < 0 || i >= static_cast<int>(terms.size())) return;
    if (terms[static_cast<std::size_t>(i)]->id > 0) req("term.kill", "term.kill", "{\"id\":" + std::to_string(terms[static_cast<std::size_t>(i)]->id) + "}");
    terms.erase(terms.begin() + i);
    term_cur = std::min(term_cur, static_cast<int>(terms.size()) - 1);
  }
  void term_send(Term& t, const std::string& data) {
    if (t.id > 0 && t.alive) req("term.write", "term.write", "{\"id\":" + std::to_string(t.id) + ",\"data\":" + q_(data) + "}");
  }

  // ---- drawing ---------------------------------------------------------------------------------------------------------------
  void hover_popup(Buffer& b, TextEditor::PopupData d) {
    if (b.ro && b.path != "@design") return ImGui::CloseCurrentPopup();
    if (b.path == "@design" || b.path.rfind("ext:", 0) == 0) return ImGui::CloseCurrentPopup();
    const std::string key = b.path + ":" + std::to_string(d.pos.line) + ":" + std::to_string(d.pos.index);
    if (key != hover_key) {
      hover_key = key, hover_valid = false;
      std::string a = "{\"path\":" + q_(b.path) + ",\"line\":" + std::to_string(d.pos.line) + ",\"col\":" + std::to_string(d.pos.index);
      if (b.dirty()) a += ",\"text\":" + q_(b.ed.GetText());
      req("prog.hover:" + key, "prog.hover", a + "}");
    }
    if (!hover_valid || hover.empty()) return ImGui::CloseCurrentPopup();
    ImGui::PushFont(nullptr, 13.f);
    ImGui::PushTextWrapPos(460.f);
    if (!hover.kind.empty()) {
      ImGui::PushStyleColor(ImGuiCol_Text, theme::kAccent);
      ImGui::Text("%s", hover.id.c_str());
      ImGui::PopStyleColor();
      ImGui::SameLine(0, 12);
      ImGui::TextDisabled("%s%s%s", hover.kind.c_str(), hover.part.empty() ? "" : "  ", hover.part.c_str());
    }
    if (!hover.markdown.empty()) ImGui::TextUnformatted(hover.markdown.c_str());
    if (!hover.kind.empty() && hover.pins.empty()) ImGui::TextDisabled("not wired to anything");
    if (!hover.pins.empty()) {
      ImGui::Separator();
      ImGui::PushFont(theme::mono(), 12.f);
      for (const PinRow& r : hover.pins) {
        ImGui::TextUnformatted((r.component + "." + r.port + (r.pin.empty() ? "" : "  [" + r.pin + "]")).c_str());
        if (r.connects.empty()) {
          ImGui::SameLine();
          ImGui::TextDisabled("  not wired");
        }
        for (const Peer& p : r.connects) {
          ImGui::SameLine();
          ImGui::TextColored(theme::kAccent, "  ->");
          ImGui::SameLine();
          ImGui::TextUnformatted((p.component + "." + p.port + (p.pin.empty() ? "" : "  [" + p.pin + "]")).c_str());
        }
      }
      ImGui::PopFont();
    }
    ImGui::PopTextWrapPos();
    ImGui::PopFont();
  }

  float find_h() const { return f_open ? (f_repl ? 68.f : 38.f) : 0.f; }
  void find_bar(Buffer* b, ImVec2 pos, float w) {
    ImGui::SetCursorScreenPos(pos);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kPanel);
    ImGui::BeginChild("##find", {w, find_h()}, false, ImGuiWindowFlags_NoScrollbar);
    ImGui::PushFont(nullptr, 13.f);
    ImGui::SetCursorPos({10, 7});
    if (ImGui::SmallButton(f_repl ? "v" : ">")) f_repl = !f_repl;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("show / hide replace");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::min(300.f, w * 0.35f));
    if (f_focus) ImGui::SetKeyboardFocusHere(), f_focus = false;
    const bool enter = ImGui::InputTextWithHint("##fq", "find", f_q, sizeof f_q, ImGuiInputTextFlags_EnterReturnsTrue);
    if (enter) {
      if (b) find_step(*b, ImGui::GetIO().KeyShift ? -1 : 1);
      ImGui::SetKeyboardFocusHere(-1);
    }
    const bool in_find = ImGui::IsItemActive();
    const auto tog = [&](const char* l, bool* v, const char* tip) {
      ImGui::SameLine();
      ImGui::PushStyleColor(ImGuiCol_Button, *v ? theme::kAccent : theme::kRaised);
      if (ImGui::SmallButton(l)) *v = !*v;
      ImGui::PopStyleColor();
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
    };
    tog("Aa", &f_case, "match case");
    tog("W", &f_word, "whole word");
    tog(".*", &f_re, "regular expression");
    ImGui::SameLine(0, 14);
    ImGui::AlignTextToFramePadding();
    if (!f_err.empty()) ImGui::TextColored(theme::kErr, "%s", f_err.c_str());
    else if (!f_q[0]) ImGui::TextDisabled("type to search");
    else if (f_m.empty()) ImGui::TextColored(theme::kErr, "no results");
    else ImGui::Text("%d of %zu%s", f_cur + 1, f_m.size(), f_m.size() >= 5000 ? "+" : "");
    ImGui::SameLine(0, 14);
    if (ImGui::SmallButton("<") && b) find_step(*b, -1);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("previous match (Shift+Enter)");
    ImGui::SameLine();
    if (ImGui::SmallButton(">") && b) find_step(*b, 1);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("next match (Enter)");
    ImGui::SameLine(w - 34);
    if (ImGui::SmallButton("x")) find_close(b);
    if (f_repl) {
      ImGui::SetCursorPos({10 + ImGui::GetFrameHeight() * 0.9f + 6, 39});
      ImGui::SetNextItemWidth(std::min(300.f, w * 0.35f));
      ImGui::InputTextWithHint("##fr", "replace", f_r, sizeof f_r);
      const bool no = !b || b->ro || f_m.empty();
      ImGui::SameLine();
      ImGui::BeginDisabled(no);
      if (ImGui::SmallButton("Replace") && b) replace_one(*b);
      ImGui::SameLine();
      if (ImGui::SmallButton("Replace all") && b) replace_all(*b);
      ImGui::EndDisabled();
      if (no && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", !b || b->ro ? "this file is read-only" : f_q[0] ? "nothing to replace: no match" : "type what to find first");
    }
    if ((in_find || (f_repl && ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows))) && ImGui::IsKeyPressed(ImGuiKey_Escape)) find_close(b);
    ImGui::PopFont();
    ImGui::EndChild();
    ImGui::PopStyleColor();
  }

  // every match of the find bar in the visible rows, over the editor (the widget has no API for this: the glyph grid is found by
  // probing its hit test, which also tracks scrolling). Not drawn while the file is scrolled sideways.
  void draw_matches(Buffer& b, ImVec2 pos, ImVec2 size) {
    TextEditor& ed = b.ed;
    if (f_m.empty() || ed.GetFirstVisibleColumn() > 0 || ed.GetLineCount() < 2 || ImGui::IsPopupOpen("##pop")) return;
    const float cw = ed.GetGlyphWidth(), lh = ed.GetLineHeight();
    const std::size_t fvr = ed.GetFirstVisibleRow(), lvr = std::min(ed.GetLastVisibleRow(), ed.GetLineCount() - 1);
    if (fvr + 2 > ed.GetLineCount()) return;
    const float mx = pos.x + size.x * 0.5f;
    float yt = -1.f;  // top of row fvr+1: the first y whose hit test says that line
    for (float y = pos.y - lh; y < pos.y + 2 * lh; y += 0.5f)
      if (ed.GetDocPosAtMousePos({mx, y}).line >= fvr + 1) {
        yt = y;
        break;
      }
    if (yt < 0) return;
    const auto top = [&](std::size_t line) { return yt + (static_cast<float>(line) - static_cast<float>(fvr + 1)) * lh; };
    float x0 = -1.f;  // left edge of column 0: the first x over a glyph on a non-empty row
    for (std::size_t l = fvr; l <= lvr && x0 < 0; ++l) {
      if (ed.GetLineText(l).empty()) continue;
      for (float x = pos.x; x < pos.x + size.x * 0.6f; x += 0.25f)
        if (ed.IsMousePosOverGlyph({x, top(l) + lh * 0.5f})) {
          x0 = x;
          break;
        }
    }
    if (x0 < 0) return;
    x0 += cw;  // the widget truncates (x - left) / cw toward zero, so column 0 also answers for the glyph-wide strip before it
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->PushClipRect(pos, {pos.x + size.x - 12, pos.y + size.y - 12}, true);
    auto it = std::lower_bound(f_m.begin(), f_m.end(), static_cast<int>(fvr), [](const FM& m, int l) { return m.line < l; });
    for (; it != f_m.end() && static_cast<std::size_t>(it->line) <= lvr; ++it) {
      const auto v0 = ed.DocPos2VisPos({static_cast<std::size_t>(it->line), static_cast<std::size_t>(it->col)}).column;
      const auto v1 = ed.DocPos2VisPos({static_cast<std::size_t>(it->line), static_cast<std::size_t>(it->col + it->len)}).column;
      const ImVec2 a{x0 + static_cast<float>(v0) * cw, top(static_cast<std::size_t>(it->line)) - 1.f}, c{x0 + static_cast<float>(v1) * cw, a.y + 16.f};  // the glyphs sit at the top of the (taller) row
      const bool cur = static_cast<int>(it - f_m.begin()) == f_cur;
      dl->AddRectFilled(a, c, cur ? IM_COL32(255, 150, 20, 120) : IM_COL32(255, 205, 60, 70), 2.f);
      if (cur) dl->AddRect(a, c, IM_COL32(255, 170, 40, 230), 2.f);
    }
    dl->PopClipRect();
  }

  void shortcuts(Buffer* b) {
    const auto chord = [](ImGuiKeyChord c) { return ImGui::IsKeyChordPressed(c, ImGuiInputFlags_None); };
    const auto lock = [](ImGuiKey k) { ImGui::SetKeyOwner(k, ImGuiKeyOwner_NoOwner, ImGuiInputFlags_LockThisFrame); };  // hide it from the widget this frame
    if (chord(ImGuiMod_Ctrl | ImGuiKey_S) && b) save(*b);
    else if (chord(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S)) save_all();
    else if (chord(ImGuiMod_Ctrl | ImGuiKey_P)) pop = Pop::Quick, pop_open = true, pop_text[0] = 0, pop_sel = 0;
    else if (chord(ImGuiMod_Ctrl | ImGuiKey_F) && b) find_open(b, false), lock(ImGuiKey_F);  // the widget's own find window never opens
    else if (chord(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_F)) focus_search(b), lock(ImGuiKey_F);
    else if (chord(ImGuiKey_F3) && b && f_open) find_step(*b, 1);
    else if (chord(ImGuiMod_Shift | ImGuiKey_F3) && b && f_open) find_step(*b, -1);
    else if (chord(ImGuiMod_Ctrl | ImGuiKey_G) && b) {
      pop = Pop::Goto, pop_open = true, pop_text[0] = 0;
      ImGui::SetKeyOwner(ImGuiKey_G, ImGuiKeyOwner_NoOwner, ImGuiInputFlags_LockThisFrame);  // the editor's own Ctrl+G is "find next"
    } else if (chord(ImGuiMod_Ctrl | ImGuiKey_H) && b) find_open(b, true);
    else if (chord(ImGuiMod_Ctrl | ImGuiKey_W)) close_tab(active, false);
    else if (chord(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_T) && !closed.empty()) {
      const std::string p = closed.back();
      closed.pop_back();
      open(p);
    } else if (chord(ImGuiKey_F12) && b) do_define(*b);
    else if (chord(ImGuiKey_F2) && b) start_rename(*b);
    else if (chord(ImGuiMod_Ctrl | ImGuiKey_GraveAccent)) {
      dock = Dock::Terminal;
      if (!terms.empty()) terms[static_cast<std::size_t>(term_cur)]->focus_input = true;
    }
  }

  bool splitter(const char* id, ImVec2 pos, ImVec2 size, bool vertical, float* delta) {
    ImGui::SetCursorScreenPos(pos);
    ImGui::InvisibleButton(id, size);
    const bool hot = ImGui::IsItemHovered() || ImGui::IsItemActive();
    if (hot) ImGui::SetMouseCursor(vertical ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
    ImGui::GetWindowDrawList()->AddRectFilled(pos, {pos.x + size.x, pos.y + size.y}, hot ? u32(theme::kAccent, 0.6f) : u32(theme::kLine));
    *delta = ImGui::IsItemActive() ? (vertical ? ImGui::GetIO().MouseDelta.x : ImGui::GetIO().MouseDelta.y) : 0.f;
    return ImGui::IsItemActive();
  }

  void explorer(ImVec2 pos, ImVec2 size) {
    ImGui::SetCursorScreenPos(pos);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kPanel);
    ImGui::BeginChild("##explorer", size, false, ImGuiWindowFlags_NoScrollbar);
    ImGui::SetCursorPos({10, 8});
    theme::caption("EXPLORER");
    ImGui::SameLine(size.x - 56);
    if (ImGui::SmallButton("+")) {
      pop = Pop::NewFile, pop_open = true, pop_text[0] = 0;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("new file");
    ImGui::SameLine();
    if (ImGui::SmallButton("R")) files_wanted = true;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("refresh the file list");
    if (!root.empty()) {
      ImGui::SetCursorPosX(10);  // flush with the caption and the tree below
      ImGui::PushStyleColor(ImGuiCol_Text, theme::kDim);
      ImGui::PushFont(theme::mono(), 12.f);
      ImGui::TextUnformatted(root.substr(root.rfind('/') == std::string::npos ? 0 : root.rfind('/') + 1).c_str());
      ImGui::PopFont();
      ImGui::PopStyleColor();
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", root.c_str());
    }
    ImGui::BeginChild("##tree", {0, 0}, false);
    ImGui::Indent(10);
    if (ImGui::Selectable("design.toml  (read-only)", cur() && cur()->path == "@design")) open("@design");
    ImGui::Separator();
    if (files_err) ImGui::TextColored(theme::kErr, "%s", files_error.c_str());
    const std::function<void(const Node&, int)> draw_node = [&](const Node& n, int depth) {
      for (const Node& k : n.kids) {
        if (!k.file) {
          ImGui::SetNextItemOpen(open_dirs.count(k.path) > 0 || (depth == 0 && open_dirs.empty() && n.kids.size() < 8), ImGuiCond_Once);
          const bool o = ImGui::TreeNodeEx(k.name.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth);
          if (o) open_dirs.insert(k.path);
          if (o) {
            draw_node(k, depth + 1);
            ImGui::TreePop();
          }
        } else {
          Buffer* b = find(k.path);
          const bool sel = cur() && cur()->path == k.path;
          ImGui::PushStyleColor(ImGuiCol_Text, ext_of(k.name) == ".py" ? theme::kText : theme::kDim);
          if (ImGui::TreeNodeEx(k.name.c_str(), ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanAvailWidth |
                                                    (sel ? ImGuiTreeNodeFlags_Selected : 0))) {
            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) open(k.path);
          }
          ImGui::PopStyleColor();
          if (b && b->dirty()) {
            ImGui::SameLine(size.x - 30);
            ImGui::TextColored(theme::kWarn, "*");
          }
        }
      }
    };
    draw_node(tree, 0);
    ImGui::EndChild();
    ImGui::EndChild();
    ImGui::PopStyleColor();
  }

  void toolbar(Buffer* b, ImVec2 pos, float w, float h) {
    ImGui::SetCursorScreenPos(pos);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kPanel);
    ImGui::BeginChild("##ptool", {w, h}, false, ImGuiWindowFlags_NoScrollbar);
    ImGui::SetCursorPos({10, (h - ImGui::GetFrameHeight()) * 0.5f});
    if (b) {
      ImGui::PushFont(theme::mono(), 12.f);
      ImGui::AlignTextToFramePadding();
      ImGui::TextColored(theme::kDim, "%s", b->path == "@design" ? "design.toml (assembly, read-only)" : b->path.c_str());
      ImGui::PopFont();
      if (b->dirty()) {
        ImGui::SameLine();
        ImGui::TextColored(theme::kWarn, "modified");
      }
      if (!b->ro) {
        ImGui::SameLine(0, 14);
        if (ImGui::SmallButton("Save")) save(*b);
      }
    } else {
      ImGui::AlignTextToFramePadding();
      ImGui::TextDisabled("Ctrl+P opens a file");
    }
    if (ImGui::GetTime() - note_t < 5.0 && !note.empty()) {
      ImGui::SameLine(0, 18);
      ImGui::AlignTextToFramePadding();
      ImGui::TextColored(note_bad ? theme::kErr : theme::kOk, "%s", note.c_str());
    }
    // "?": the shortcuts (the status bar carries no hints)
    ImGui::SetCursorPos({w - ProgramUi::kRunSlotW - 30, (h - ImGui::GetFrameHeight()) * 0.5f});
    if (ImGui::SmallButton("?")) ImGui::OpenPopup("##shortcuts");
    ImGui::SetItemTooltip("keyboard shortcuts");
    if (ImGui::BeginPopup("##shortcuts")) {
      theme::caption("SHORTCUTS");
      static const char* kKeys[][2] = {{"Ctrl+P", "open a file"}, {"Ctrl+S / Ctrl+Shift+S", "save / save all"}, {"Ctrl+F / Ctrl+H", "find / replace"},
                                       {"Ctrl+G", "go to line"}, {"Ctrl+Shift+F", "find in files"}, {"F12 / F2", "definition / rename (python)"},
                                       {"Ctrl+W / Ctrl+Shift+T", "close / reopen tab"}, {"Ctrl+`", "terminal"}, {"Ctrl+L", "clear the terminal"}};
      for (const auto& k : kKeys) {
        ImGui::TextColored(theme::kDim, "%-22s", k[0]);
        ImGui::SameLine();
        ImGui::TextUnformatted(k[1]);
      }
      ImGui::EndPopup();
    }
    // ---- RUN SLOT (top-right of the editor): the RUN controls (Run / Test + bench|full toggle) draw here ----
    ImGui::SetCursorPos({w - ProgramUi::kRunSlotW, 0});
    ImGui::BeginChild("##runslot", {ProgramUi::kRunSlotW, h}, false, ImGuiWindowFlags_NoScrollbar);
    if (self->run_slot) {
      ImGui::SetCursorPos({0, (h - ImGui::GetFrameHeight()) * 0.5f});
      self->run_slot({ProgramUi::kRunSlotW, h});
    } else {
      ImGui::SetCursorPos({ProgramUi::kRunSlotW - 78, (h - ImGui::GetFrameHeight()) * 0.5f});
      ImGui::BeginDisabled();
      theme::button("Run", false, {68, 0});
      ImGui::EndDisabled();
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Run / Test controls arrive with the RUN backend");
    }
    ImGui::EndChild();
    ImGui::EndChild();
    ImGui::PopStyleColor();
  }

  void editor_tabs(ImVec2 pos, float w, float h) {
    ImGui::SetCursorScreenPos(pos);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kBar);
    ImGui::BeginChild("##etabs", {w, h}, false, ImGuiWindowFlags_NoScrollbar);
    if (ImGui::BeginTabBar("##bufs", ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_FittingPolicyScroll | ImGuiTabBarFlags_AutoSelectNewTabs)) {
      int close_req = -1;
      for (int i = 0; i < static_cast<int>(bufs.size()); ++i) {
        Buffer& b = *bufs[static_cast<std::size_t>(i)];
        bool keep = true;
        ImGuiTabItemFlags f = (b.dirty() ? ImGuiTabItemFlags_UnsavedDocument : 0) | (want_tab == i ? ImGuiTabItemFlags_SetSelected : 0);
        const std::string lab = b.label + "###" + b.path;
        if (ImGui::BeginTabItem(lab.c_str(), &keep, f)) {
          active = i;
          ImGui::EndTabItem();
        }
        if (!keep) close_req = i;
      }
      want_tab = -1;
      ImGui::EndTabBar();
      if (close_req >= 0) close_tab(close_req, false);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
  }

  // scripted hover (screenshots, tests): put the mouse over the glyph (hover_line, hover_col) by probing the editor's hit test
  void hold_mouse(Buffer& b, ImVec2 pos, ImVec2 size) {
    const float cw = b.ed.GetGlyphWidth(), lh = b.ed.GetLineHeight();
    float fy = -1.f;
    for (float y = pos.y + 2; y < pos.y + size.y && fy < 0; y += lh * 0.5f)
      if (b.ed.GetDocPosAtMousePos({pos.x + size.x * 0.5f, y}).line == static_cast<std::size_t>(hover_line)) fy = y;
    if (fy < 0) return;
    for (float x = pos.x; x < pos.x + size.x - 20; x += cw * 0.5f)
      if (b.ed.IsMousePosOverGlyph({x, fy}) && b.ed.GetDocPosAtMousePos({x, fy}).index == static_cast<std::size_t>(hover_col)) {
        ImGui::GetIO().AddMousePosEvent(x + 1, fy);
        return;
      }
  }

  void editor_body(Buffer* b, ImVec2 pos, ImVec2 size) {
    ImGui::SetCursorScreenPos(pos);
    if (!b) {
      ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4{0.106f, 0.110f, 0.122f, 1.f});
      ImGui::BeginChild("##welcome", size, false, ImGuiWindowFlags_NoScrollbar);
      ImGui::SetCursorPos({size.x * 0.5f - 150, size.y * 0.35f});
      ImGui::BeginGroup();
      theme::caption("PROGRAM");
      ImGui::TextDisabled("Ctrl+P            open a file");
      ImGui::TextDisabled("Ctrl+Shift+F      find in files");
      ImGui::TextDisabled("Ctrl+`            terminal");
      ImGui::TextDisabled("F12 / F2          definition / rename (python)");
      ImGui::EndGroup();
      ImGui::EndChild();
      ImGui::PopStyleColor();
      return;
    }
    if (!b->loaded) {
      ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4{0.106f, 0.110f, 0.122f, 1.f});
      ImGui::BeginChild("##loading", size, false);
      if (b->error.empty()) ImGui::TextDisabled("loading %s ...", b->path.c_str());
      else ImGui::TextColored(theme::kErr, "%s", b->error.c_str());
      ImGui::EndChild();
      ImGui::PopStyleColor();
      return;
    }
    ImGui::PushFont(theme::mono(), 14.f);
    b->ed.Render(("##ed" + b->path).c_str(), size);
    ImGui::PopFont();
    if (hover_on) hold_mouse(*b, pos, size);
    if (f_open) draw_matches(*b, pos, size);
    if (b->ext_scroll >= 0.f && b->ed.GetLineCount() > 1) {
      b->ed.ScrollToLine(static_cast<std::size_t>(b->ext_scroll * static_cast<float>(b->ed.GetLineCount() - 1)), TextEditor::Scroll::alignTop);
      b->ext_scroll = -1.f;
    }
    if (b->check_due && !b->check_flight && !b->ro) {
      const std::string e = ext_of(b->path);
      b->check_due = false;
      if (e == ".py" || e == ".toml" || e == ".json") {
        b->check_flight = true;
        req("prog.check:" + b->path, "prog.check", "{\"path\":" + q_(b->path) + ",\"text\":" + q_(b->ed.GetText()) + "}");
      }
    }
  }

  // ---- the terminal view -----------------------------------------------------------------------------------------------------
  static int hist_cb(ImGuiInputTextCallbackData* d) {
    Term* t = static_cast<Term*>(d->UserData);
    t->input_sel = d->HasSelection();
    if (d->EventFlag == ImGuiInputTextFlags_CallbackHistory && !t->history.empty()) {
      const int n = static_cast<int>(t->history.size());
      t->hist_pos = std::clamp(t->hist_pos + (d->EventKey == ImGuiKey_UpArrow ? -1 : 1), 0, n);
      d->DeleteChars(0, d->BufTextLen);
      if (t->hist_pos < n) d->InsertChars(0, t->history[static_cast<std::size_t>(t->hist_pos)].c_str());
    }
    return 0;
  }

  void term_view(Term& t, ImVec2 size, bool input) {
    ImGui::PushFont(theme::mono(), 13.f);
    const float lh = ImGui::GetTextLineHeightWithSpacing(), cw = ImGui::CalcTextSize("M").x, fs = ImGui::GetFontSize();
    ImFont* font = ImGui::GetFont();
    const float in_h = input ? ImGui::GetFrameHeight() + 10 : 0;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4{0.082f, 0.086f, 0.097f, 1.f});
    ImGui::BeginChild("##tout", {size.x, size.y - in_h}, false, ImGuiWindowFlags_HorizontalScrollbar);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 base = ImGui::GetCursorScreenPos();
    base.x += 8;
    const int n = static_cast<int>(t.lines.size());
    ImGuiListClipper clip;
    clip.Begin(n, lh);
    while (clip.Step()) {
      for (int r = clip.DisplayStart; r < clip.DisplayEnd; ++r) {
        const TermLine& l = t.lines[static_cast<std::size_t>(r)];
        const float y = base.y + static_cast<float>(r) * lh;
        if (t.sel_on) {
          int r0 = t.sr0, c0 = t.sc0, r1 = t.sr1, c1 = t.sc1;
          if (r0 > r1 || (r0 == r1 && c0 > c1)) std::swap(r0, r1), std::swap(c0, c1);
          if (r >= r0 && r <= r1) {
            const int a = r == r0 ? c0 : 0, b = r == r1 ? c1 + 1 : static_cast<int>(std::max<std::size_t>(l.ch.size(), 1));
            dl->AddRectFilled({base.x + static_cast<float>(a) * cw, y}, {base.x + static_cast<float>(b) * cw, y + lh}, u32(theme::kAccent, 0.35f));
          }
        }
        std::size_t i = 0;
        while (i < l.ch.size()) {
          std::size_t j = i;
          std::string s;
          while (j < l.ch.size() && l.at[j] == l.at[i]) put_utf8(s, l.ch[j]), ++j;
          const ImVec2 p{base.x + static_cast<float>(i) * cw, y};
          const ImU32 col = ansi_color(l.at[i]);
          dl->AddText(font, fs, p, col, s.c_str(), s.c_str() + s.size());
          if (l.at[i] & kBold) dl->AddText(font, fs, {p.x + 0.7f, p.y}, col, s.c_str(), s.c_str() + s.size());
          i = j;
        }
      }
    }
    ImGui::SetCursorScreenPos(base);
    ImGui::Dummy({static_cast<float>(t.max_cols + 2) * cw, static_cast<float>(n) * lh});
    // selection with the mouse
    if (ImGui::IsWindowHovered()) {
      const ImVec2 m = ImGui::GetMousePos();
      const int row = std::clamp(static_cast<int>(std::floor((m.y - base.y) / lh)), 0, n - 1);
      const int col = std::max(0, static_cast<int>(std::floor((m.x - base.x) / cw)));
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) t.dragging = true, t.sel_on = false, t.sr0 = t.sr1 = row, t.sc0 = t.sc1 = col;
      else if (t.dragging && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        t.sr1 = row, t.sc1 = col;
        t.sel_on = (row != t.sr0 || col != t.sc0);
      }
      if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && row < n) {
        const std::u32string& s = t.lines[static_cast<std::size_t>(row)].ch;
        int a = std::min<int>(col, static_cast<int>(s.size())), b = a;
        while (a > 0 && s[static_cast<std::size_t>(a) - 1] != U' ') --a;
        while (b < static_cast<int>(s.size()) && s[static_cast<std::size_t>(b)] != U' ') ++b;
        t.sr0 = t.sr1 = row, t.sc0 = a, t.sc1 = b - 1, t.sel_on = b > a;
      }
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) ImGui::OpenPopup("##tctx");
      if (t.id == -1 && !input && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !t.sel_on && row < n) {  // OUTPUT: a traceback line opens its file
        std::string line;
        for (char32_t c : t.lines[static_cast<std::size_t>(row)].ch) put_utf8(line, c);
        static const std::regex kTb(R"re(File "([^"]+)", line (\d+))re");
        std::smatch m;
        if (std::regex_search(line, m, kTb)) {
          std::string p = m[1];
          if (!root.empty() && p.rfind(root + "/", 0) == 0) p = p.substr(root.size() + 1);  // absolute path under the program root
          if (!p.empty() && p[0] != '/') open(p, std::stoi(m[2]) - 1, 0);
        }
      }
    }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) t.dragging = false;
    if (ImGui::BeginPopup("##tctx")) {
      if (ImGui::MenuItem("Copy", "Ctrl+C", false, t.sel_on)) ImGui::SetClipboardText(t.selected_text().c_str());
      if (ImGui::MenuItem("Paste", "Ctrl+V", false, input)) {
        const char* c = ImGui::GetClipboardText();
        if (c) std::strncat(t.input, c, sizeof t.input - std::strlen(t.input) - 1);
      }
      if (ImGui::MenuItem("Clear")) t.lines.assign(1, TermLine{}), t.cx = 0, t.sel_on = false;
      ImGui::EndPopup();
    }
    if (t.stick && t.new_data) ImGui::SetScrollHereY(1.f);
    t.stick = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.f;
    t.new_data = false;
    ImGui::EndChild();
    ImGui::PopStyleColor();
    if (input) {
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);
      if (!t.alive) {
        ImGui::TextDisabled("process exited - close this tab or open a new terminal (+)");
      } else {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(theme::kAccent, ">");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-8);
        if (t.focus_input) ImGui::SetKeyboardFocusHere(), t.focus_input = false;
        const bool enter = ImGui::InputText("##tin", t.input, sizeof t.input,
                                            ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory | ImGuiInputTextFlags_CallbackAlways, hist_cb, &t);
        const bool active = ImGui::IsItemActive() || enter;
        if (enter) {
          const std::string line = t.input;
          if (!line.empty() && (t.history.empty() || t.history.back() != line)) t.history.push_back(line);
          t.hist_pos = static_cast<int>(t.history.size());
          term_send(t, line + "\n");
          t.input[0] = 0;
          t.stick = true;
          ImGui::SetKeyboardFocusHere(-1);
        }
        if (active) {
          const ImGuiIO& io = ImGui::GetIO();
          if (io.KeyCtrl && !io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_C, false) && !t.input_sel) {
            if (t.sel_on) ImGui::SetClipboardText(t.selected_text().c_str());
            else term_send(t, "\x03"), t.input[0] = 0;  // the tty's line discipline turns it into SIGINT for the foreground job
          } else if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_D, false) && !t.input[0]) term_send(t, "\x04");
          else if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_L, false)) t.lines.assign(1, TermLine{}), t.cx = 0, t.sel_on = false;
          else if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z, false)) term_send(t, "\x1a");
        }
      }
    }
    ImGui::PopFont();
  }

  void dock_area(ImVec2 pos, ImVec2 size, const std::deque<std::string>& log) {
    ImGui::SetCursorScreenPos(pos);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kBar);
    ImGui::BeginChild("##dock", size, false, ImGuiWindowFlags_NoScrollbar);
    std::size_t nprob = 0;
    for (auto& b : bufs) nprob += b->problems.size();
    const std::string pl = "PROBLEMS" + (nprob ? " " + std::to_string(nprob) : ""), sl = "SEARCH" + (matches.empty() ? "" : " " + std::to_string(matches.size()));
    const char* labels[] = {"TERMINAL", pl.c_str(), sl.c_str(), "OUTPUT", "LOG"};
    ImGui::SetCursorPos({12, 2});
    const int c = theme::tabs(labels, 5, static_cast<int>(dock), 22.f);
    if (c >= 0) dock = static_cast<Dock>(c);
    ImGui::SetCursorPos({0, 32});
    const ImVec2 inner{size.x, size.y - 32};
    switch (dock) {
      case Dock::Terminal: {
        if (terms.empty()) {
          ImGui::SetCursorPos({12, 40});
          if (ImGui::Button("Open a terminal")) new_term();
          break;
        }
        ImGui::SetCursorPos({10, 32});
        std::vector<std::string> names;
        for (auto& t : terms) names.push_back(t->title + (t->alive ? "" : " (exited)"));
        std::vector<const char*> lp;
        for (auto& s : names) lp.push_back(s.c_str());
        const int cc = theme::tabs(lp.data(), static_cast<int>(lp.size()), term_cur, 14.f);
        if (cc >= 0) term_cur = cc, terms[static_cast<std::size_t>(cc)]->focus_input = true;
        ImGui::SameLine();
        if (ImGui::SmallButton("+")) new_term();
        ImGui::SameLine();
        if (ImGui::SmallButton("x")) kill_term(term_cur);
        if (!terms.empty()) {
          ImGui::SetCursorPos({0, 62});
          term_view(*terms[static_cast<std::size_t>(std::min(term_cur, static_cast<int>(terms.size()) - 1))], {inner.x, inner.y - 30}, true);
        }
        break;
      }
      case Dock::Problems: {
        ImGui::SetCursorPos({0, 32});
        ImGui::BeginChild("##probs", {inner.x, inner.y}, false);
        bool any = false;
        for (auto& b : bufs)
          for (const Problem& p : b->problems) {
            any = true;
            ImGui::TextColored(theme::kErr, "x");
            ImGui::SameLine();
            const std::string row = b->path + ":" + std::to_string(p.line + 1) + ":" + std::to_string(p.col + 1) + "   " + p.msg;
            if (ImGui::Selectable(row.c_str())) open(b->path, p.line, p.col);
          }
        if (!any) ImGui::TextDisabled("no problems (python, TOML and JSON syntax is checked as you type)");
        ImGui::EndChild();
        break;
      }
      case Dock::Search: {
        ImGui::SetCursorPos({10, 34});
        ImGui::SetNextItemWidth(std::min(360.f, inner.x - 260));
        if (q_focus) ImGui::SetKeyboardFocusHere(), q_focus = false;
        const bool enter = ImGui::InputTextWithHint("##q", "find in files", q, sizeof q, ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        const auto tog = [&](const char* l, bool* v, const char* tip) {
          ImGui::PushStyleColor(ImGuiCol_Button, *v ? theme::kAccent : theme::kRaised);
          if (ImGui::SmallButton(l)) *v = !*v;
          ImGui::PopStyleColor();
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
          ImGui::SameLine();
        };
        tog("Aa", &q_case, "match case");
        tog("W", &q_word, "whole word");
        tog(".*", &q_regex, "regular expression");
        if ((ImGui::SmallButton("Search") || enter) && q[0]) search_files();
        // as you type: once the query / options have been still for 250 ms (and on open, when a query was prefilled)
        const std::string key = q_key();
        if (key != q_seen) q_seen = key, q_t = ImGui::GetTime();
        if (!q[0]) {
          if (!matches.empty()) matches.clear();
          q_sent.clear();
        } else if (key != q_sent && !q_flight && ImGui::GetTime() - q_t > 0.25) {
          search_files();
        }
        ImGui::SameLine(0, 16);
        {
          std::set<std::string> fs;
          for (const Match& m : matches) fs.insert(m.path);
          if (q[0] && key == q_sent && !q_flight) ImGui::TextDisabled("%zu result(s) in %zu file(s)", matches.size(), fs.size());
          else if (q[0]) ImGui::TextDisabled("searching ...");
        }
        ImGui::SetCursorPos({0, 62});
        ImGui::BeginChild("##res", {inner.x, inner.y - 30}, false);
        if (!q_error.empty()) ImGui::TextColored(theme::kErr, "%s", q_error.c_str());
        for (const Match& m : matches) {
          ImGui::PushFont(theme::mono(), 13.f);
          const std::string row = m.path + ":" + std::to_string(m.line + 1) + "   " + m.text;
          if (ImGui::Selectable(row.c_str())) open(m.path, m.line, m.col);
          ImGui::PopFont();
        }
        if (truncated) ImGui::TextDisabled("(showing the first %zu)", matches.size());
        ImGui::EndChild();
        break;
      }
      case Dock::Output: {
        ImGui::SetCursorPos({0, 32});
        term_view(*output, inner, false);
        break;
      }
      case Dock::Log: {
        ImGui::SetCursorPos({0, 32});
        ImGui::BeginChild("##log", {inner.x, inner.y}, false);
        ImGui::PushFont(theme::mono(), 13.f);
        for (const std::string& l : log) ImGui::TextUnformatted(l.c_str());
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4) ImGui::SetScrollHereY(1.f);
        ImGui::PopFont();
        ImGui::EndChild();
        break;
      }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
  }

  void popups(Buffer* b, ImVec2 anchor) {
    // (opened here, not where the key / button is: a popup's id depends on the window it is opened in)
    if (pop_open) ImGui::OpenPopup("##pop"), pop_open = false;
    if (discard_req) ImGui::OpenPopup("##discard"), discard_req = false;
    ImGui::SetNextWindowPos(anchor, ImGuiCond_Always, {0.5f, 0.f});
    ImGui::SetNextWindowSizeConstraints({420, 0}, {560, 460});
    if (ImGui::BeginPopup("##pop", ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize)) {
      const char* hint = pop == Pop::Quick ? "open file (type to filter)" : pop == Pop::Goto ? "go to line" : pop == Pop::Rename ? "new name" : "new file (path relative to the root)";
      if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
      ImGui::SetNextItemWidth(-1);
      const bool enter = ImGui::InputTextWithHint("##pt", hint, pop_text, sizeof pop_text, ImGuiInputTextFlags_EnterReturnsTrue);
      if (pop == Pop::Quick) {
        std::vector<std::pair<int, const std::string*>> hits;
        for (const std::string& f : files)
          if (const int s = fuzzy(f, pop_text); s >= 0) hits.emplace_back(s, &f);
        std::stable_sort(hits.begin(), hits.end(), [](auto& a, auto& b2) { return a.first < b2.first; });
        const int n = static_cast<int>(std::min<std::size_t>(hits.size(), 12));
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) pop_sel = std::min(pop_sel + 1, n - 1);
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) pop_sel = std::max(pop_sel - 1, 0);
        pop_sel = std::clamp(pop_sel, 0, std::max(n - 1, 0));
        std::string pick;
        for (int i = 0; i < n; ++i)
          if (ImGui::Selectable(hits[static_cast<std::size_t>(i)].second->c_str(), i == pop_sel)) pick = *hits[static_cast<std::size_t>(i)].second;
        if (enter && n > 0) pick = *hits[static_cast<std::size_t>(pop_sel)].second;
        if (!pick.empty()) open(pick), ImGui::CloseCurrentPopup();
      } else if (enter) {
        if (pop == Pop::Goto && b) {
          const int ln = std::atoi(pop_text);
          if (ln >= 1) {
            b->ed.SetCursor(TextEditor::DocPos(static_cast<std::size_t>(ln - 1), 0));
            b->ed.ScrollToLine(static_cast<std::size_t>(ln - 1), TextEditor::Scroll::alignMiddle);
            b->ed.SetFocus();
          }
        } else if (pop == Pop::Rename && b && pop_text[0]) {
          run_rename(*b, pop_text);
        } else if (pop == Pop::NewFile && pop_text[0]) {
          req("prog.write:" + std::string(pop_text), "prog.write", "{\"path\":" + q_(pop_text) + ",\"text\":\"\"}");
          open(pop_text);
          files_wanted = true;
        }
        ImGui::CloseCurrentPopup();
      }
      if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
      ImGui::EndPopup();
    }
    if (ImGui::BeginPopupModal("##discard", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {
      const std::string nm = close_ask >= 0 && close_ask < static_cast<int>(bufs.size()) ? bufs[static_cast<std::size_t>(close_ask)]->label : "";
      ImGui::Text("Save changes to %s before closing?", nm.c_str());
      if (theme::button("Save", true)) {
        if (close_ask >= 0 && close_ask < static_cast<int>(bufs.size())) save(*bufs[static_cast<std::size_t>(close_ask)]);
        close_tab(close_ask, true);
        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      if (ImGui::Button("Don't save")) close_tab(close_ask, true), ImGui::CloseCurrentPopup();
      ImGui::SameLine();
      if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
      ImGui::EndPopup();
    }
  }
};

// ---- ProgramUi ----------------------------------------------------------------------------------------------------------------
ProgramUi::ProgramUi() : s_(std::make_unique<State>(send)) { s_->self = this; }
ProgramUi::~ProgramUi() = default;

void ProgramUi::draw(const std::deque<std::string>& log, bool has_backend) {
  State& s = *s_;
  if (!has_backend) {
    ImGui::TextDisabled("(no design editor attached)");
    return;
  }
  if (!s.term_started) s.term_started = true, s.new_term();
  const ImVec2 org = ImGui::GetWindowPos(), sz = ImGui::GetWindowSize();
  Buffer* b = s.cur();
  s.shortcuts(b);
  s.exp_w = std::clamp(s.exp_w, 140.f, sz.x * 0.4f);
  s.bot_h = std::clamp(s.bot_h, 80.f, sz.y - 200.f);
  const float sp = 4.f, ex = s.exp_w;
  s.explorer(org, {ex, sz.y});
  float d = 0;
  s.splitter("##sv", {org.x + ex, org.y}, {sp, sz.y}, true, &d);
  s.exp_w += d;
  const float x0 = org.x + ex + sp, w = sz.x - ex - sp, th = 36.f, tabs_h = 30.f;
  s.toolbar(b, {x0, org.y}, w, th);
  s.editor_tabs({x0, org.y + th}, w, tabs_h);
  b = s.cur();
  if (b) s.find_update(*b);
  const float fh = s.find_h(), body_h = sz.y - th - tabs_h - fh - s.bot_h - sp;
  if (s.f_open) s.find_bar(b, {x0, org.y + th + tabs_h}, w);
  s.editor_body(b, {x0, org.y + th + tabs_h + fh}, {w, body_h});
  s.splitter("##sh", {x0, org.y + th + tabs_h + fh + body_h}, {w, sp}, false, &d);
  s.bot_h -= d;
  s.dock_area({x0, org.y + sz.y - s.bot_h}, {w, s.bot_h}, log);
  s.popups(s.cur(), {x0 + w * 0.5f, org.y + th + tabs_h + 10});
}

void ProgramUi::on_reply(const std::string& kind, const Json& r) {
  State& s = *s_;
  const std::string tool = tool_of(kind), tag = tag_of(kind);
  const bool ok = r.at("ok").num != 0;
  const std::string err = r.s("error");
  const Json& res = r.at("result");
  if (tool == "prog.files") {
    s.files_flight = false;
    s.files_err = !ok;
    s.files_error = err;
    if (ok) {
      s.root = res.s("root");
      s.files.clear();
      for (const Json& f : res.at("files").arr) s.files.push_back(f.str);
      s.rebuild_tree();
    }
  } else if (tool == "prog.read") {
    Buffer* b = s.find(tag);
    if (!b) return;
    if (!ok) return void(b->error = err);
    b->ed.SetText(res.s("text"));
    b->saved_undo = static_cast<int>(b->ed.GetUndoIndex());
    b->loaded = true;
    b->check_due = true;
    if (b->jump_line >= 0) s.jump(*b);
    else b->ed.SetFocus();
  } else if (tool == "prog.design") {
    Buffer* b = s.find("@design");
    if (!b || !ok) return;
    b->ed.SetReadOnlyEnabled(false);
    b->ed.SetText(res.s("text"));
    b->ed.SetReadOnlyEnabled(true);
    b->ed.SetLanguage(toml_language());
    b->loaded = true;
  } else if (tool == "prog.write") {
    Buffer* b = s.find(tag);
    if (!ok) return s.say("save failed: " + err, true);
    if (b) b->saved_undo = b->save_idx, b->check_due = true;
    s.say("saved " + tag);
    if (std::find(s.files.begin(), s.files.end(), tag) == s.files.end()) s.files_wanted = true;
  } else if (tool == "prog.check") {
    Buffer* b = s.find(tag);
    if (!b) return;
    b->check_flight = false;
    if (!ok) return;
    b->problems.clear();
    b->ed.ClearMarkers();
    for (const Json& p : res.at("problems").arr) {
      b->problems.push_back({static_cast<int>(p.n("line")), static_cast<int>(p.n("col")), static_cast<int>(p.n("end_col")), p.s("message")});
      b->ed.AddMarker(static_cast<std::size_t>(p.n("line")), IM_COL32(230, 70, 70, 255), IM_COL32(220, 50, 50, 70), p.s("message"), p.s("message"));
    }
  } else if (tool == "prog.search") {
    s.q_flight = false;
    s.q_error = ok ? "" : err;
    s.matches.clear();
    if (ok) {
      for (const Json& m : res.at("matches").arr) s.matches.push_back({m.s("path"), m.s("text"), static_cast<int>(m.n("line")), static_cast<int>(m.n("col"))});
      s.truncated = res.at("truncated").num != 0;
    }
  } else if (tool == "prog.define") {
    if (!ok) return s.say(err, true);
    if (!res.s("path").empty()) s.open(res.s("path"), static_cast<int>(res.n("line")), static_cast<int>(res.n("col")));
    else if (!res.s("external").empty()) s.say(res.s("external"));
    else s.say("no definition found", true);
  } else if (tool == "prog.rename") {
    if (!ok) return s.say("rename: " + err, true);
    std::map<std::string, std::vector<const Json*>> by;
    for (const Json& e : res.at("edits").arr) by[e.s("path")].push_back(&e);
    for (auto& [path, es] : by) {
      Buffer* b = s.find(path);
      if (!b || !b->loaded) continue;
      std::stable_sort(es.begin(), es.end(), [](const Json* a, const Json* c) { return a->n("line") != c->n("line") ? a->n("line") > c->n("line") : a->n("col") > c->n("col"); });
      for (const Json* e : es)  // bottom-up: earlier positions stay valid; every edit is one undo step of that buffer
        b->ed.ReplaceSectionText(TextEditor::DocPos(static_cast<std::size_t>(e->n("line")), static_cast<std::size_t>(e->n("col"))),
                                 TextEditor::DocPos(static_cast<std::size_t>(e->n("end_line")), static_cast<std::size_t>(e->n("end_col"))), e->s("text"));
      b->saved_undo = static_cast<int>(b->ed.GetUndoIndex());  // the files were written by the backend: the buffers are clean (undo re-dirties)
    }
    s.say("renamed in " + std::to_string(by.size()) + " file(s)");
  } else if (tool == "prog.hover") {
    if (tag != s.hover_key) return;
    s.hover = Hover{};
    if (ok) {
      const Json& en = res.at("entity");
      s.hover.kind = en.s("kind"), s.hover.id = en.s("id"), s.hover.part = en.s("part"), s.hover.markdown = res.s("markdown");
      for (const Json& p : res.at("pins").arr) {
        PinRow row{p.s("component"), p.s("port"), p.s("pin"), {}};
        for (const Json& c : p.at("connects").arr) row.connects.push_back({c.s("component"), c.s("port"), c.s("pin")});
        s.hover.pins.push_back(std::move(row));
      }
    }
    s.hover_valid = true;
  } else if (tool == "term.open") {
    for (auto& t : s.terms)
      if (t->key == tag) {
        if (ok) t->id = static_cast<int>(res.n("id"));
        else t->alive = false, t->feed("\r\n[could not start a shell: " + err + "]\r\n");
      }
  } else if (tool == "term.read") {
    for (auto& t : s.terms)
      if (t->id == std::atoi(tag.c_str())) {
        if (ok && !res.s("data").empty()) t->feed(res.s("data"));
        if (!ok || res.at("alive").num == 0) {
          if (!t->exited_noted) t->exited_noted = true, t->feed("\r\n\x1b[2m[process exited]\x1b[0m\r\n");
          t->alive = false;
        }
      }
  } else if (!ok && tool != "term.write" && tool != "term.kill") {
    s.say(tool + ": " + err, true);
  }
}

bool ProgramUi::poll(double now, bool page_visible, std::string* kind, std::string* json) {
  State& s = *s_;
  if (s.files_wanted && page_visible && !s.files_flight) {
    s.files_wanted = false, s.files_flight = true;
    *kind = "prog.files", *json = design::request("prog.files");
    return true;
  }
  const double every = page_visible ? 0.07 : 0.5;  // terminals keep draining when the page is not shown (a chatty job must not block)
  for (std::size_t k = 0; k < s.terms.size(); ++k) {
    Term& t = *s.terms[(s.poll_rr + k) % s.terms.size()];
    if (t.id <= 0 || !t.alive || now - t.last_poll < every) continue;
    t.last_poll = now;
    s.poll_rr = (s.poll_rr + k + 1) % s.terms.size();
    *kind = "term.read:" + std::to_string(t.id);
    *json = design::request("term.read", "{\"id\":" + std::to_string(t.id) + "}");
    return true;
  }
  return false;
}

void ProgramUi::show_external(const std::string& name, const std::string& text, float scroll) {
  State& s = *s_;
  if (name.empty()) {
    s.open("@design");
    return;
  }
  const std::string path = "ext:" + name;
  Buffer* b = s.find(path);
  if (!b) {
    s.bufs.push_back(std::make_unique<Buffer>());
    b = s.bufs.back().get();
    b->path = path, b->label = base_of(name), b->ro = true;
    s.configure(*b);
    b->ed.SetLanguage(lang_for(name));
    b->ed.SetText(text);
    b->ed.SetReadOnlyEnabled(true);
    b->loaded = true;
  }
  b->ext_scroll = scroll;
  s.active = s.want_tab = static_cast<int>(std::find_if(s.bufs.begin(), s.bufs.end(), [&](auto& x) { return x.get() == b; }) - s.bufs.begin());
}

void ProgramUi::command(const std::string& c) {
  State& s = *s_;
  Buffer* b = s.cur();
  if (c == "save" && b) s.save(*b);
  else if (c == "save_all") s.save_all();
  else if (c == "find" && b) s.find_open(b, false);
  else if (c == "replace" && b) s.find_open(b, true);
  else if (c == "goto_line" && b) s.pop = State::Pop::Goto, s.pop_open = true, s.pop_text[0] = 0;
  else if (c == "quick_open") s.pop = State::Pop::Quick, s.pop_open = true, s.pop_text[0] = 0, s.pop_sel = 0;
  else if (c == "find_in_files") s.focus_search(b);
  else if (c == "new_terminal") s.new_term();
  else if (c == "clear_terminal" && !s.terms.empty()) {
    Term& t = *s.terms[static_cast<std::size_t>(s.term_cur)];
    t.lines.assign(1, TermLine{}), t.cx = 0, t.sel_on = false;
  } else if (c == "close_tab") s.close_tab(s.active, false);
  else if (c == "reopen_tab" && !s.closed.empty()) {
    const std::string p = s.closed.back();
    s.closed.pop_back();
    s.open(p);
  } else if (c == "toggle_comment" && b) b->ed.ToggleComments();
  else if (c == "design") s.open("@design");
}

void ProgramUi::script(const Json& e) {
  State& s = *s_;
  const std::string cmd = e.s("cmd");
  Buffer* b = s.cur();
  if (cmd == "prog_open") {
    s.open(e.s("path"), e.at("line").kind == Json::Num ? static_cast<int>(e.n("line")) : -1, static_cast<int>(e.n("col")), static_cast<int>(e.n("len")));
    s.hover_on = false;
  } else if (cmd == "prog_type" && b && b->loaded) b->ed.ReplaceTextInCurrentCursor(e.s("text"));
  else if (cmd == "prog_cmd") command(e.s("name"));
  else if (cmd == "prog_term" && !s.terms.empty()) s.term_send(*s.terms[static_cast<std::size_t>(s.term_cur)], e.s("text") + "\n");
  else if (cmd == "prog_hover") s.hover_on = e.n("line", -1) >= 0, s.hover_line = static_cast<int>(e.n("line")), s.hover_col = static_cast<int>(e.n("col"));
  else if (cmd == "prog_find" && b) {  // type into the find (and replace) fields
    s.find_open(b, e.at("replace").kind == Json::Str);
    std::snprintf(s.f_q, sizeof s.f_q, "%s", e.s("text").c_str());
    if (e.at("replace").kind == Json::Str) std::snprintf(s.f_r, sizeof s.f_r, "%s", e.s("replace").c_str());
    s.f_case = e.at("case").num != 0, s.f_word = e.at("word").num != 0, s.f_re = e.at("regex").num != 0;
    s.f_key.clear(), s.f_goto = true, s.f_focus = false;
  } else if (cmd == "prog_find_next" && b) s.find_step(*b, 1);
  else if (cmd == "prog_find_prev" && b) s.find_step(*b, -1);
  else if (cmd == "prog_replace" && b) s.replace_one(*b);
  else if (cmd == "prog_replace_all" && b) s.replace_all(*b);
  else if (cmd == "prog_search") {  // find in files: type the query (send:false = leave it to the as-you-type search)
    s.dock = Dock::Search;
    std::snprintf(s.q, sizeof s.q, "%s", e.s("text").c_str());
    s.q_case = e.at("case").num != 0, s.q_word = e.at("word").num != 0, s.q_regex = e.at("regex").num != 0;
    if (e.at("send").kind != Json::Bool || e.at("send").num != 0) s.search_files();
  } else if (cmd == "prog_goto" && b && b->loaded) {
    const std::size_t l = static_cast<std::size_t>(std::max(1.0, e.n("line")) - 1);
    b->ed.SetCursor(TextEditor::DocPos(l, 0));
    b->ed.ScrollToLine(l, TextEditor::Scroll::alignMiddle);
    b->ed.SetFocus();
  } else if (cmd == "prog_define" && b) s.do_define(*b);
  else if (cmd == "prog_rename" && b) s.run_rename(*b, e.s("new"));
  else if (cmd == "prog_quick_open") {  // the palette with `text` typed; pick = open the best match instead of showing it
    s.pop_sel = 0;
    std::snprintf(s.pop_text, sizeof s.pop_text, "%s", e.s("text").c_str());
    const std::string* best = nullptr;
    int bs = 1 << 30;
    for (const std::string& f : s.files)
      if (const int sc = fuzzy(f, s.pop_text); sc >= 0 && sc < bs) bs = sc, best = &f;
    if (e.at("pick").num != 0 && best) s.open(*best);
    else s.pop = State::Pop::Quick, s.pop_open = true;
  } else if (cmd == "prog_dock") {
    const std::string n = e.s("name");
    s.dock = n == "problems" ? Dock::Problems : n == "search" ? Dock::Search : n == "output" ? Dock::Output : n == "log" ? Dock::Log : Dock::Terminal;
  }
}

void ProgramUi::shutdown() {
  State& s = *s_;
  for (auto& t : s.terms)
    if (t->id > 0) send("term.kill", design::request("term.kill", "{\"id\":" + std::to_string(t->id) + "}"));
}

void ProgramUi::save_all() { s_->save_all(); }
std::string ProgramUi::active_path() const { return s_->cur() && s_->cur()->path.rfind("ext:", 0) != 0 && s_->cur()->path != "@design" ? s_->cur()->path : ""; }
void ProgramUi::output_append(const std::string& text) { s_->output->feed(text); }
void ProgramUi::output_clear() {
  s_->output->lines.assign(1, TermLine{});
  s_->output->cx = 0;
}
void ProgramUi::show_output() { s_->dock = Dock::Output; }
void ProgramUi::open_file(const std::string& path, int line, int col) { s_->open(path, line, col); }
int ProgramUi::open_buffers() const { return static_cast<int>(s_->bufs.size()); }
bool ProgramUi::any_dirty() const {
  for (auto& b : s_->bufs)
    if (b->dirty()) return true;
  return false;
}
int ProgramUi::terminal_count() const { return static_cast<int>(s_->terms.size()); }

}  // namespace progui
