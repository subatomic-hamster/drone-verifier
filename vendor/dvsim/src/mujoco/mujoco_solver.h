#pragma once
#include <mujoco/mujoco.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "core/solver.h"
#include "mujoco/devices/devices.h"

class MujocoSolver final : public Solver {
 public:
  // Takes ownership of m and d; builds the prim registry. `spec` (shared, never mutated: structural edits work on a
  // copy) is what Spawn / Delete edit; without one, `spec_xml` (replay-log header) is parsed on the first edit.
  MujocoSolver(mjModel* m, mjData* d, std::shared_ptr<mjSpec> spec = nullptr, std::string spec_xml = {});
  ~MujocoSolver() override;

  double dt() const override { return m_->opt.timestep; }
  int step(int n) override;
  bool diverged() const override { return diverged_; }
  std::vector<Prim> prims() const override { return prims_; }
  Result get(const std::string& prim, const std::string& attr, const std::string& frame,
             std::vector<double>& out) const override;
  Result set(const std::string& prim, const std::string& attr, const std::string& frame,
             std::span<const double> values) override;
  std::string unit(const std::string& prim, const std::string& attr) const override;
  std::vector<std::byte> save() const override;
  void restore(std::span<const std::byte> bytes) override;
  std::unique_ptr<Solver> clone() const override;
  std::uint64_t state_hash() const override;
  Weather weather() const override;
  std::vector<GeomVisual> visuals() const override;
  std::vector<GeomPose> poses() const override;
  std::vector<TubeVisual> tubes() const override { return collect_tubes(nullptr); }
  std::vector<std::pair<std::string, std::string>> tube_ends() const override {
    std::vector<std::pair<std::string, std::string>> ends;
    collect_tubes(&ends);
    return ends;
  }
  std::vector<double> raycast(std::span<const double> origin, std::span<const double> directions) const override;
  Result spawn(const std::string& model_path, const std::string& at, std::span<const double> pos,
               std::span<const double> quat) override;
  Result remove(const std::string& prim) override;
  std::vector<std::byte> model_bytes() const override;
  std::string version() const override { return std::string("mujoco ") + mj_versionString(); }

  const mjModel& model() const { return *m_; }
  const mjData& data() const { return *d_; }
  const devs::Devices* devices() const { return dev_.get(); }  // null: the model declares no dvsim.devices (tests)

 private:
  struct Ref { enum Kind { body, joint, actuator, sensor, device } kind; int id; };  // device: id = slot in dev_->slots
  struct Loc { double* p; int n; bool read_only; };
  bool locate(const Ref& r, const std::string& attr, Loc& loc) const;
  // The cables (tubes()); with `ends`, their end instances too (tube_ends()): the tendon is named "wire:<i>:<c>|<end a>|<end b>".
  std::vector<TubeVisual> collect_tubes(std::vector<std::pair<std::string, std::string>>* ends) const;
  bool is_servo(int a) const;  // actuator a is a position servo: kp/kv are settable
  void build_index();          // prims_ / index_ / structure_ from the model
  std::shared_ptr<mjSpec> working_spec();  // private compiled copy, synced to the model's parameters; throws
  // Runs `edit` on a spec copy, recompiles it against copies of the model and data (state kept by mj_recompile), and
  // only then swaps everything in: on any failure nothing has changed.
  Result restructure(const std::function<Result(mjSpec*)>& edit);

  mjModel* m_;
  mjData* d_;
  std::shared_ptr<mjSpec> spec_;  // null until needed in a solver rebuilt from a replay-log blob
  std::string spec_xml_;
  std::unique_ptr<devs::Devices> dev_;  // from the model's "dvsim.devices" custom text; null = plain mj_step
  bool diverged_ = false;
  std::uint64_t structure_ = 0;  // hash of the prim registry + sizes: a snapshot only restores onto the same structure
  std::vector<Prim> prims_;
  std::unordered_map<std::string, Ref> index_;
};

// Installs the MuJoCo error/warning handlers (idempotent). Errors throw std::runtime_error.
void install_mujoco_handlers();
// World::Loader for MJCF/URDF. Throws std::runtime_error with MuJoCo's message on failure.
std::unique_ptr<Solver> load_mujoco(const std::string& path);
// Same, from an mj_saveModel buffer (replay-log header). Throws std::runtime_error on failure.
std::unique_ptr<Solver> load_mujoco_bytes(std::span<const std::byte> bytes);
