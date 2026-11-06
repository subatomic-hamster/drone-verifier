#pragma once
// The component library: search box, category tree, param filters, thumbnail grid, detail panel, and the compact list used as the
// BUILD sidebar. Data comes from the editing backend (library.search / categories / ranges / info, python/dvsim/design/uilib.py);
// this file only draws it and turns clicks into `Actions` for viewer.cpp. Cards are drag sources: payload "DVSIM_PART" = part id.
#include <imgui.h>

#include <map>
#include <string>
#include <vector>

#include "viewer/build3d.h"

namespace libui {

struct Hit {
  std::string id, name, vendor, category, thumb;
  bool has_mass = false;
  double mass = 0;
  bool mine = false;  // a MY PARTS card (a workspace sub-assembly file): id = its path; the drag payload is "DVSIM_MYPART"
};
// MY PARTS: an assembly file with an [exports] table (uilib.my_parts).
struct MyPart {
  std::string name, path, exports;  // exports: "2 ports, 1 connector"
  bool has_mass = false, has_cost = false;
  double mass = 0, cost = 0;
};
// The category row key of MY PARTS in Model::category.
inline constexpr const char* kMyParts = "@my";
struct Cat {
  std::string path;
  int count = 0, depth = 0;
};
struct Range {
  double lo = 0, hi = 0;
  std::string unit;
};
struct ParamRow {
  std::string name, unit, fidelity, src, url, quote, value;
};
struct PortRow {
  std::string name, kind, type, gender;
};
struct Info {
  std::string id, name, vendor, category, part_number, url, thumb;
  std::vector<ParamRow> params;
  std::vector<PortRow> ports;
  std::vector<b3::Box> boxes;  // part frame
};
struct Filter {
  bool on = false;
  float lo = 0, hi = 0;
};

struct Model {
  char query[96] = "";
  std::string category;
  std::map<std::string, Filter> filters;  // param / alias -> [lo, hi]
  std::vector<Hit> hits;
  int total = 0;
  std::vector<Cat> cats;
  std::map<std::string, Range> ranges;
  std::map<std::string, Info> infos;
  std::map<std::string, MyPart> mine;  // path -> row, filled by library.my_parts
  int my_count = 0;                    // number of MY PARTS (the category row's count)
  std::string selected;     // the part shown in the detail panel
  bool dirty = true;        // the query / category / filters changed: ask the backend again
  std::string ranges_for;   // category the ranges were computed for
  // scripted typing (library.search with type_ms): the box fills one character per type_ms
  std::string type_target;
  double type_ms = 0, type_next = 0;
  std::map<std::string, ImVec2> card_at;  // last drawn screen centre of each part's card (scripted drags start there)
  std::string filters_json() const;
};

struct Actions {
  bool search = false;          // send library.search with the current query / filters
  std::string info;             // request library.info for this part
  std::string add;              // ADD button: add this part to the selection (or the best open port)
  std::string add_mine;         // ADD on a MY PARTS card: insert this sub-assembly file
  std::string swap;             // SWAP button: swap the selected instance for this part
  std::string open_page;        // a card double-clicked in the sidebar: show it on the LIBRARY page
};

// Decoded PNG thumbnails (build/thumbs/<id>.png, tools/thumbs.py) as ImGui textures; nothing found -> a category placeholder.
class Thumbs {
 public:
  ~Thumbs();
  ImTextureRef get(const std::string& path);  // empty ref when the file cannot be decoded
 private:
  std::map<std::string, ImTextureData*> tex_;
};

void tick_typing(Model& m, double now, Actions& a);
// Full page: filters | grid | detail. `selection` = the selected instance id (for ADD / SWAP labels), `targets` = how many open
// compatible ports the detail part could snap to (-1 unknown).
void page(Model& m, Thumbs& th, ImVec2 pos, ImVec2 size, const std::string& selection, int targets, Actions& a);
// Compact list for the BUILD sidebar: search + category chips + small cards.
void sidebar(Model& m, Thumbs& th, ImVec2 pos, ImVec2 size, Actions& a);
// Placeholder / thumbnail drawn in a rect (also used by the drag ghost chip).
void thumb_rect(Thumbs& th, const Hit& h, ImVec2 a, ImVec2 b);

}  // namespace libui
