#pragma once
// The ONE solver interface (seam). dvsim_core must never include a solver's headers.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

struct Prim {
  std::string path;  // "/world/<body>[/<child>...]", "/world/<body>/<joint>", "/actuators/<n>", "/sensors/<n>"
  std::string kind;  // "body" | "joint" | "actuator" | "sensor"
};

// Outcome of get/set; the bus maps it to Reply.error.
struct Result {
  enum Code { ok, not_found, invalid_argument } code = ok;
  std::string message;
};

// Static look of one geom (what the renderer tessellates once). visuals() and poses() are index-parallel.
// size: plane = half extents x,y (<= 0: unbounded); sphere = r; capsule = r, half length (along local z);
// ellipsoid = semi-axes; cylinder = r, half length; box = half extents. `other` (SDF, ...) is not drawn. hfield: the heightfield already
// triangulated into `mesh` (like mesh, geom frame; shaded as terrain by the renderer, never part of the shadow-map fit); size = its
// half extents x, y and elevation range.
enum class GeomType { plane, sphere, capsule, ellipsoid, cylinder, box, mesh, other, hfield };
struct GeomVisual {
  GeomType type = GeomType::other;
  double size[3] = {0, 0, 0};
  float rgba[4] = {0.5f, 0.5f, 0.5f, 1};
  int group = 0;            // renderers hide group 3 only (collision-only geoms); 4 = design builder's vehicle geoms
  // MuJoCo material (defaults = MuJoCo's own): specular strength, shininess (x128 = Blinn-Phong exponent); metallic /
  // roughness < 0 = unset (then only specular / shininess shade the highlight).
  float specular = 0.5f, shininess = 0.5f, metallic = -1, roughness = -1;
  std::vector<float> mesh;  // mesh only: unrolled triangles, per vertex position xyz then normal xyz (geom frame)
  // The sea surface (a Sea device's wave field sampled on a grid at the current time): drawn as water (Fresnel sky reflection, sun glitter,
  // depth tint), never casts a shadow. The vertex count is fixed for a scene, so a renderer overwrites the mesh in place frame by frame.
  // Marked by group (the solver sets it on the sea mesh), not by a member: GeomVisual keeps its size.
  static constexpr int kWaterGroup = 6;
  bool water() const { return group == kWaterGroup; }
};
// A cable drawn as a tube through world points (MuJoCo spatial tendon through sites). `length` = the cable's real
// length (tendon resting length; <= the point path => taut); the renderer sags the slack. pts: xyz triples, world, double.
struct TubeVisual {
  double radius = 0.001, length = 0;
  float rgba[4] = {0.5f, 0.5f, 0.5f, 1};
  std::vector<double> pts;
};
// World pose of a geom frame: position in DOUBLE (the renderer subtracts the camera in double) + row-major 3x3 rotation.
struct GeomPose {
  double pos[3];
  double rot[9];
};

// The weather a renderer shows (a pure read of the scene's rain device; all zero = the clear look, unchanged): rain rate mm/h,
// the wind the drops drift with (m/s, world), sim time s (so streaks move with the simulation, not the wall clock).
struct Weather {
  double rain = 0, wind[3] = {0, 0, 0}, time = 0;
};

class Solver {
 public:
  virtual ~Solver() = default;
  virtual Weather weather() const { return {}; }

  virtual double dt() const = 0;
  // Advance up to n steps; stops early on divergence. Returns steps actually taken.
  virtual int step(int n) = 0;
  virtual bool diverged() const = 0;  // sticky

  virtual std::vector<Prim> prims() const = 0;
  // frame: "" or "world", or a body prim path. SI units; attribute table lives in the solver.
  virtual Result get(const std::string& prim, const std::string& attr, const std::string& frame,
                     std::vector<double>& out) const = 0;
  virtual Result set(const std::string& prim, const std::string& attr, const std::string& frame,
                     std::span<const double> values) = 0;
  virtual std::string unit(const std::string& prim, const std::string& attr) const = 0;  // "" if unknown

  virtual std::vector<std::byte> save() const = 0;
  virtual void restore(std::span<const std::byte> bytes) = 0;  // throws on size mismatch
  virtual std::unique_ptr<Solver> clone() const = 0;
  virtual std::uint64_t state_hash() const = 0;

  // Rendering / sensing seam. Pure reads: they change no state (derived kinematics only, never hashed).
  virtual std::vector<GeomVisual> visuals() const = 0;
  virtual std::vector<GeomPose> poses() const = 0;
  virtual std::vector<TubeVisual> tubes() const { return {}; }  // cables (visual only, follow their bodies)
  // The instance ids at the two ends of each cable of tubes(), index-parallel ("" = unknown): the viewer hides a cable by them.
  // (A separate list, not TubeVisual members: new members there made Sim.tubes() crash in the Python module, cause not found.)
  virtual std::vector<std::pair<std::string, std::string>> tube_ends() const { return {}; }
  // Ray distances in metres from `origin` (3 doubles) along each direction (3 doubles each, any nonzero length; the
  // solver normalises). -1 = no hit. Deterministic. Nothing is excluded.
  virtual std::vector<double> raycast(std::span<const double> origin, std::span<const double> directions) const = 0;

  // Structural edits (Spawn / Delete commands). Existing entities keep their state; prims() changes; a snapshot taken
  // before the edit makes restore() throw. `pos` / `quat`: empty or 3 / 4 values. On failure nothing has changed.
  virtual Result spawn(const std::string& model_path, const std::string& at, std::span<const double> pos,
                       std::span<const double> quat) = 0;
  virtual Result remove(const std::string& prim) = 0;

  // Replay-log header support: the loaded model as a self-contained buffer (a BytesLoader rebuilds a
  // solver from it) and a version string; replay refuses logs whose version differs.
  virtual std::vector<std::byte> model_bytes() const = 0;
  virtual std::string version() const = 0;
};
