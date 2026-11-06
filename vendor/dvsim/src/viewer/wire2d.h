#pragma once
// The 2D "flatten" wiring canvas. Components picked in the Design window are shown as cards (name + the chosen terminals as
// labelled, colour-coded pins); dragging pin -> pin wires them. The viewer holds no wiring logic: every action is a
// `flatten.*` design_api request (python/dvsim/design/wire2d.py, listed there) and the canvas draws the state the replies and
// the rev poll bring back -- so a scripted agent driving the same ops through dvsim.design.viewer.Backend sees the UI react:
// cards pop in, a wire is drawn with a drag animation.
#include <imgui.h>

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "viewer/design.h"

namespace wire2d {

// The far end of a wire at a pin: component, its port and (a connector pin) the pin; formatted by wire2d.py (`pin_links`).
struct Link {
  std::string component, port, pin;
};
// One drawn terminal: a port of the part (conn empty) or one real connector pin (name = "CONN.PIN", label = "num name").
struct Pin {
  std::string name, kind, type, gender, role, side;  // side: "left" | "right"; kind: power | ground | signal | nc
  int row = 0;
  bool wired = false;
  std::string label, conn, tip, flag;  // flag: "" | "warning" | "error" (the pin DRC rules)
  std::vector<std::string> flag_msgs;
  std::string port;           // the part's port this terminal belongs to
  std::vector<Link> connects;  // what its wires lead to (the tooltip)
};
// The box around the pins of one connector (side "both" = a two-row header: row 0 on the left edge, row 1 on the right).
struct ConnBox {
  std::string id, name, type, side;
  int row = 0, nrows = 1;
};
struct Card {
  std::string id, part, group;
  float x = 0, y = 0, w = 200, h = 60;  // canvas units (x, y = where it is drawn now; tx, ty = where the layout put it)
  float x0 = 0, y0 = 0, tx = 0, ty = 0;
  std::vector<Pin> pins;
  std::vector<ConnBox> conns;
  double born = 0;  // ImGui time the card pops in
};
// One conductor of a wire: pin to pin.
struct Strand {
  std::string from, to, flag;
  std::vector<ImVec2> route;  // canvas units: orthogonal polyline from the `from` pin to the `to` pin (the backend routes it, wire2d_route.py)
};
struct Wire {
  std::string name, from, to, part, kind, flag;
  double awg = 0, length = 0;
  int conductors = 1;
  double born = -1e9;  // ImGui time its draw animation starts (far past: no animation)
  std::vector<Strand> strands;  // one per conductor; strands[0] carries the label
  std::vector<std::string> flag_msgs;
  std::string label;          // gauge text and where it goes (canvas centre); empty = no free spot
  ImVec2 label_at{0, 0}, label_size{0, 0};
};

class Canvas {
 public:
  // send(tool, args_json): queues one design_api request (the viewer's request queue); replies come back to on_reply.
  using Send = std::function<void(const std::string& tool, const std::string& args, bool quiet)>;
  explicit Canvas(Send send) : send_(std::move(send)) {}

  bool active() const { return active_; }
  // Hover cards / hover highlight of the canvas: 0 auto (only while a person moves the mouse), 1 always, 2 never (ui.view tooltips).
  void set_tooltips(int mode) { tip_mode_ = mode; }
  // Ribbon hooks (WIRING page): flatten the picked components, or the BUILD selection when nothing is picked; zoom to 100 % (every
  // pin label readable); fit the sheet in the window.
  void begin_selected(const design::State& st, const std::string& selected);
  void zoom_to_pins();
  void set_side_open(bool open) { side_open_ = open; }
  // ui.wire_pick: tick exactly these components in the picker (all their power / signal terminals).
  void pick(const std::vector<std::string>& ids, const design::State& st);
  void fit_view() { fit(); }
  // The picker's AUTOWIRE section (folded behind a button in its action row): `autowire_ui` draws the body, `autowire_n` is the number of
  // decisions waiting for the user (shown on the button).
  std::function<void()> autowire_ui;
  int autowire_n = 0;
  // ui.wire_view: fit the sheet, or zoom to `zoom` (0 = keep) with canvas point (cx, cy) at the middle of the sheet (NaN = keep the
  // current middle; kKeep = 1e30); `seconds` > 0 glides there (eased) instead of jumping: a recorded pan / zoom.
  static constexpr double kKeep = 1e30;
  void view(double zoom, bool fit_it, double cx = kKeep, double cy = kKeep, double seconds = 0);
  // A message for the canvas's toast strip (a refusal of an op a scripted agent sent: the window never saw that reply).
  void notify(const std::string& text, bool bad) { toast(text, bad); }
  // Inside the viewer's WIRING pane window (cursor at the pane's top-left, `size` = its content area): the component /
  // terminal picker, or the flattened canvas + side panel. `selected`: the instance selected in the viewer ("" = none).
  // `hidden`: the instances switched off with the eye in the browser: their cards and every wire ending on them are not drawn.
  void page(ImVec2 size, const design::State& st, const std::string& selected, const std::set<std::string>& hidden);
  // The zoom of the canvas in percent (shown in the top bar), 0 while the picker is up.
  int zoom_percent() const { return active_ ? static_cast<int>(zoom_ * 100 + 0.5f) : 0; }
  // The reply (parsed JSON) of a flatten.* request, failed ones included.
  void on_reply(const std::string& tool, const design::Json& reply);
  // From the viewer's rev poll: the backend's canvas revision; a change fetches the state.
  void poll(int flat_rev);

 private:
  struct Drag {
    enum Kind { none, on_pin, on_card } kind = none;
    std::string card, pin;  // pin drag: where it started
    float dx = 0, dy = 0;   // card drag: grab offset (canvas units)
  };
  static Card parse_card(const design::Json& c);
  void apply(const design::Json& f);
  void select_ui(const design::State& st, const std::string& selected);
  void preview_ui(ImVec2 at, ImVec2 size);
  // One card in screen space: origin (ox, oy) + canvas * zoom, popped to scale `s` (0..1). `bare`: the picker's preview (only the
  // component name, connector boxes and pin dots).
  struct CardView {
    float ox, oy, zoom, s;
    bool bare;
    const std::map<std::string, int>* ends;  // conductors ending on each pin ("card.pin"): 2+ get a junction dot
    const Pin* hover;
    float fade = 1;  // 0..1 opacity (the canvas fading in)
  };
  void draw_card(ImDrawList* dl, const Card& c, const CardView& v) const;
  float side_w() const;
  bool hidden(const std::string& id) const { return hid_ && hid_->count(id) > 0; }
  bool wire_hidden(const Wire& w) const;
  std::string pick_args() const;
  void begin_flatten();
  void canvas_ui();
  void side_ui();
  void fit();
  const Card* card(const std::string& id) const;
  bool pin_pos(const Card& c, const Pin& p, float& x, float& y) const;
  void toast(const std::string& s, bool bad);
  // The wire's screen polyline (corners rounded). While one of its cards is being dragged the backend's route is stale: a plain
  // Z-route between the two pins is drawn until the card is dropped and the backend routes again.
  std::vector<ImVec2> wire_path(const Wire& w, ImVec2* label_c, bool* has_label) const;
  std::vector<ImVec2> strand_path(const Strand& s) const;  // the same for one conductor (no label)
  std::string strand_kind(const Wire& w, const Strand& s) const;
  // Top-left key of the canvas: two stacked panels (symbols, controls), each folds up with a click on its title. Returns the
  // screen rectangle both occupy (the mouse over it does not reach the sheet).
  ImVec4 legend(ImDrawList* dl, ImVec2 org, float a);

  Send send_;
  bool active_ = false;            // the backend's flatten mode
  double alpha_ = 0;               // 0..1, eased: the canvas overlay fades in / out
  int seen_rev_ = -1;
  std::vector<Card> cards_;
  std::vector<Wire> wires_;
  int hidden_wires_ = 0;
  std::string sel_wire_, hover_wire_;
  Drag drag_;
  ImVec2 org_{0, 0}, sz_{800, 600};  // the pane (screen coordinates), set by page()
  float zoom_ = 1, ox_ = 40, oy_ = 40;  // canvas -> screen: screen = o + canvas * zoom
  struct ViewAnim {                      // ui.wire_view with seconds: an eased glide of (zoom, ox, oy)
    double t0 = -1, dur = 0;
    float z0 = 1, ox0 = 0, oy0 = 0, z1 = 1, ox1 = 0, oy1 = 0;
  } va_;
  bool fit_pending_ = false;
  std::string toast_;
  bool toast_bad_ = false;
  double toast_t_ = -1e9;
  double layout_t_ = -1e9;         // when the backend last moved the cards: they glide from (x0, y0) to (tx, ty), wires fade in after
  double new_wire_slot_ = 0;       // staggers the draw animations of wires that arrive together
  int tip_mode_ = 0;
  double move_t_ = -1e9;           // last time the mouse moved
  const std::set<std::string>* hid_ = nullptr;  // the viewer's hidden instances, set by page()
  bool side_open_ = true;                       // the right panel of the canvas (remembered while the viewer runs)
  bool legend_open_ = true;                     // the symbols key
  bool autowire_open_ = false;                  // the picker's AUTOWIRE section
  ImVec4 legend_rect_{0, 0, 0, 0};              // x0 y0 x1 y1 of the key as drawn last frame
  // selection (3D mode)
  std::map<std::string, std::set<std::string>> pick_;  // instance -> chosen terminals
  std::vector<Card> preview_;                          // the picked components as cards (flatten.preview), drawn right of the list
  std::string preview_sig_, preview_asked_;            // what preview_ shows / was last asked for
  char filter_[64] = "";
  char len_buf_[32] = "";
  int awg_idx_ = 5;
};

}  // namespace wire2d
