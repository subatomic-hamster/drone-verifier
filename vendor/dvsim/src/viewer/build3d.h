#pragma once
// The BUILD mode's 3D interaction layer, drawn with ImGui over the scene (the renderer draws only the model): projection of
// world points / pixel rays with the renderer's own camera maths, picking against per-geom oriented boxes, the transform
// gizmo (axis arrows + ground-plane handle, rotation rings) with grid / angle / edge / port snapping and smart guides, selection
// brackets, and the drag-and-drop ghost. Pure functions + small state structs; viewer.cpp owns the design state and decides what
// a finished drag means (a `gizmo.move` / `gizmo.rotate` / `library.drop` request to the editing backend).
#include <imgui.h>

#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/render.h"

namespace b3 {

struct V3 {
  double x = 0, y = 0, z = 0;
  double& operator[](int i) { return i == 0 ? x : i == 1 ? y : z; }
  double operator[](int i) const { return i == 0 ? x : i == 1 ? y : z; }
};
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double len(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 unit(V3 a) { const double l = len(a); return l > 0 ? a * (1 / l) : a; }
inline V3 axis_vec(int i) { V3 v; v[i] = 1; return v; }

struct Mat3 {  // row-major
  double m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  V3 operator*(V3 v) const { return {m[0] * v.x + m[1] * v.y + m[2] * v.z, m[3] * v.x + m[4] * v.y + m[5] * v.z, m[6] * v.x + m[7] * v.y + m[8] * v.z}; }
  V3 col(int j) const { return {m[j], m[3 + j], m[6 + j]}; }
  Mat3 operator*(const Mat3& o) const {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) r.m[3 * i + j] = m[3 * i] * o.m[j] + m[3 * i + 1] * o.m[3 + j] + m[3 * i + 2] * o.m[6 + j];
    return r;
  }
  Mat3 T() const { Mat3 r; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) r.m[3 * i + j] = m[3 * j + i]; return r; }
};
Mat3 rot_axis(V3 axis, double angle);
// URDF fixed-axis rpy (R = Rz Ry Rx) <-> matrix, as the Python side (design/snap.py rpy_matrix / matrix_rpy).
Mat3 rpy_to_mat(V3 rpy);
V3 mat_to_rpy(const Mat3& R);

struct Box {  // oriented box: centre, rotation (columns = box axes), half extents
  V3 c;
  Mat3 R;
  V3 h;
};
struct Port {
  std::string name;
  V3 pos, axis;
  bool used = false;
};
struct InstGeo {
  std::string id, part;
  V3 pos, lo, hi;  // origin of the part frame + world AABB of its geometry
  Mat3 R;
  std::vector<Box> boxes;
  std::vector<Port> ports;
  bool free = false;  // free-placed (pos / rpy in the document) rather than held by a mate
  bool root = false;
  bool has_color = false;
  float color[4] = {0, 0, 0, 1};
};
using Scene = std::map<std::string, InstGeo>;

struct Ray {
  V3 o, d;
};
struct Cam {
  V3 pos, r, u, f;
  double sx = 1, t = 1, w = 1, h = 1;  // clip scale, 1/tan(fovy/2), framebuffer size
  static Cam make(const Camera& c, double w, double h);
  bool project(V3 p, ImVec2& out) const;  // false: behind the camera
  Ray ray(ImVec2 px) const;
  double px_per_m(V3 p) const;  // screen pixels per metre at p (perpendicular to the view)
};

struct Hit {
  std::string id;
  double t = 1e300;
  V3 point, normal;
};
bool ray_box(const Ray& r, const Box& b, double& t, V3* normal = nullptr);
// Nearest instance box under the pixel (nullopt-style: id empty when nothing). skip: an instance to ignore (the one being moved).
// hidden: instances switched off in the browser (not pickable).
Hit pick(const Cam& cam, ImVec2 px, const Scene& scene, const std::string& skip = "", const std::set<std::string>* hidden = nullptr);
bool ray_plane(const Ray& r, V3 p0, V3 n, V3& hit);

struct Guide {
  int axis = 0;  // 0 x / 1 y / 2 z
  V3 a, b;       // the two aligned feature points (on the moving part / on the other part)
  bool port = false;
};
struct Settings {
  double grid = 0.005;  // m; <= 0 off
  double angle = 15;    // deg; <= 0 off
  bool edges = true, ports = true, guides = true;
};

// Smart-guide snapping of a world translation of `id` (same rules as design/uiops.py snap_move): its AABB edges / centre planes
// align to those of the other instances (|gap| < thresh m), its mech ports land on other instances' ports, then the origin goes
// to the grid on the axes nothing else snapped. `mask` = the axes that may change. Returns the snapped delta and the guides.
V3 snap_translate(const Scene& scene, const InstGeo& moving, V3 delta, const bool mask[3], double thresh, const Settings& s,
                  std::vector<Guide>& guides);

struct GizmoOut {
  bool hot = false;       // the mouse is over a handle or dragging one: the caller must not orbit / pick
  bool dragging = false;
  bool released = false;  // the drag ended this frame with a non-zero change: send it
  bool rotate = false;
  V3 delta;               // translation (world)
  V3 axis;                // rotation axis (world), angle in degrees
  double angle = 0;
  std::vector<Guide> guides;
};
class Gizmo {
 public:
  enum Mode { Move, Rotate, Off };
  Mode mode = Move;
  // Handles for `target` (draws them on the background draw list, clipped to `clip`). `ok`: the pointer may start a drag
  // (viewport hovered, nothing else active). `invert_snap` (Ctrl held): flips grid / angle snapping for this drag.
  GizmoOut update(const Cam& cam, const Scene& scene, const InstGeo& target, const Settings& s, bool ok, bool invert_snap, ImVec4 clip);
  void cancel() { drag_ = -1; }
  bool dragging() const { return drag_ >= 0; }

 private:
  int hover_ = -1, drag_ = -1;  // handle ids: 0..2 axes / rings, 3 ground plane
  double t0_ = 0, a0_ = 0;
  V3 p0_, h0_;
  double moved_ = 0;
};

// ---- drawing (all on the background draw list, so panels cover them) ----
ImU32 axis_color(int axis, float alpha = 1);
void draw_box(ImDrawList* dl, const Cam& cam, const Box& b, ImU32 col, float th = 1.f);
void draw_brackets(ImDrawList* dl, const Cam& cam, V3 lo, V3 hi, ImU32 col, float th = 1.5f);
void draw_guides(ImDrawList* dl, const Cam& cam, const std::vector<Guide>& g, float alpha);
// The ghost of a part (its local boxes at a pose): wire + faint fill.
void draw_ghost(ImDrawList* dl, const Cam& cam, const std::vector<Box>& local, V3 pos, const Mat3& R, ImU32 col);
void draw_ring(ImDrawList* dl, const Cam& cam, V3 centre, V3 axis, double radius_m, ImU32 col, float th);
// Box of `b` after the rigid motion x -> pos' + Rd (x - pos)
Box moved_box(const Box& b, V3 pivot, const Mat3& Rd, V3 shift);

}  // namespace b3
