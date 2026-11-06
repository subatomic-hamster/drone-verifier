#include "core/shm.h"

#include <bit>
#include <cstring>
#include <new>
#include <stdexcept>

#include "dvsim_generated.h"

namespace shm = dvsim::shm;

namespace {
// Runs one command on the bus; returns the reply's dvsim.Err (0 = ok).
std::uint32_t run(World& w, const Session& s, flatbuffers::FlatBufferBuilder& f, dvsim::Command t,
                  flatbuffers::Offset<void> cmd) {
  f.Finish(dvsim::CreateEnvelope(f, 0, f.CreateString("shm"), t, cmd));
  const flatbuffers::DetachedBuffer b = w.execute({f.GetBufferPointer(), f.GetSize()}, s);
  return static_cast<std::uint32_t>(flatbuffers::GetRoot<dvsim::Reply>(b.data())->error());
}
}  // namespace

ShmServer::ShmServer(World& w, const Session& session, const std::string& name, Obs obs, std::uint32_t max_steps)
    : w_(w), session_(session), name_(name), obs_(std::move(obs)), max_steps_(max_steps) {
  for (const Prim& p : w_.solver().prims())
    if (p.kind == "actuator") actuators_.push_back(p.path);
  std::uint64_t nobs = 0;
  std::vector<double> v;
  for (const auto& [prim, attr] : obs_) {  // validate before the segment exists
    const Result r = w_.solver().get(prim, attr, "", v);
    if (r.code != Result::ok) throw std::runtime_error("dvsim shm obs " + prim + "." + attr + ": " + r.message);
    nobs += v.size();
  }
  const std::uint64_t nu = actuators_.size(), total = sizeof(shm::Header) + 8 * (nu + nobs);
  if (total > shm::kMaxSegment) throw std::runtime_error("dvsim shm: block too large");

  const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0)
    throw std::runtime_error("dvsim shm '" + name + "': shm_open: " + std::strerror(errno) +
                             (errno == EEXIST ? " (leftover segment? remove /dev/shm" + name + ")" : ""));
  void* p = MAP_FAILED;
  if (ftruncate(fd, static_cast<off_t>(total)) == 0) p = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  const int err = errno;
  close(fd);
  if (p == MAP_FAILED) {
    shm_unlink(name.c_str());
    throw std::runtime_error("dvsim shm '" + name + "': " + std::strerror(err));
  }
  size_ = total;
  h_ = new (p) shm::Header{};
  std::memcpy(h_->magic, shm::kMagic, sizeof h_->magic);
  h_->version = shm::kVersion;
  h_->header_size = sizeof(shm::Header);
  h_->nu = static_cast<std::uint32_t>(nu);
  h_->nobs = static_cast<std::uint32_t>(nobs);
  h_->total_size = total;
  for (std::size_t i = 0; i < nu; ++i) {  // ctrl[] starts at the world's current ctrl, so round(0) changes nothing
    w_.solver().get(actuators_[i], "ctrl", "", v);
    reinterpret_cast<double*>(shm::ctrl_base(h_))[i] = v.at(0);
  }
  round(0);  // initial obs / tick / hash, so a client may read them before its first step
  h_->ready.store(1, std::memory_order_release);
}

ShmServer::~ShmServer() {
  h_->closed.store(1, std::memory_order_release);
  h_->resp_seq.fetch_add(1);  // changes the word a client sleeps on (no lost wake-up), then wake it
  shm::wake(h_->resp_seq);
  munmap(h_, size_);
  shm_unlink(name_.c_str());
}

// Applies ctrl (changed values only) and steps n ticks, then publishes status/tick/hash/obs. Request fields are
// copied out of the segment once: the client may scribble on them at any time.
void ShmServer::round(std::uint32_t n) {
  std::uint32_t status = 0;
  if (n > max_steps_) {
    status = shm::kBadRequest;
  } else {
    const std::span<const double> shared(reinterpret_cast<const double*>(shm::ctrl_base(h_)), h_->nu);
    const std::vector<double> ctrl(shared.begin(), shared.end());
    std::vector<double> cur;
    for (std::size_t i = 0; i < ctrl.size() && status == 0; ++i) {
      w_.solver().get(actuators_[i], "ctrl", "", cur);
      if (cur.size() == 1 && std::bit_cast<std::uint64_t>(cur[0]) == std::bit_cast<std::uint64_t>(ctrl[i])) continue;
      flatbuffers::FlatBufferBuilder f;
      const std::vector<double> one{ctrl[i]};
      status = run(w_, session_, f, dvsim::Command::SetAttr,
                   dvsim::CreateSetAttrDirect(f, actuators_[i].c_str(), "ctrl", "", &one).Union());
    }
    if (status == 0 && n > 0) {
      flatbuffers::FlatBufferBuilder f;
      status = run(w_, session_, f, dvsim::Command::Step, dvsim::CreateStep(f, static_cast<int>(n)).Union());
    }
  }
  double* out = reinterpret_cast<double*>(shm::ctrl_base(h_)) + h_->nu;
  std::uint32_t at = 0;
  std::vector<double> v;
  for (const auto& [prim, attr] : obs_) {
    if (w_.solver().get(prim, attr, "", v).code != Result::ok || at + v.size() > h_->nobs) {
      status = status ? status : static_cast<std::uint32_t>(dvsim::Err::solver_error);
      break;
    }
    std::memcpy(out + at, v.data(), v.size() * sizeof(double));
    at += static_cast<std::uint32_t>(v.size());
  }
  h_->status = status;
  h_->tick = w_.tick();
  h_->state_hash = w_.state_hash();
}

std::uint64_t ShmServer::serve(double idle_timeout_s) {
  std::uint64_t rounds = 0;
  for (;;) {
    // served_ (not the current req_seq): a client that posted before serve() started must still be answered.
    if (!shm::wait_ne(h_->req_seq, served_, shm::after(idle_timeout_s)))
      throw std::runtime_error("dvsim shm '" + name_ + "': no request for " + std::to_string(idle_timeout_s) +
                               " s (client gone?)");
    const std::uint32_t served = served_ = h_->req_seq.load(std::memory_order_acquire);
    const std::uint32_t cmd = h_->cmd, n = h_->n_steps;
    if (cmd == shm::kStep) {
      round(n);
      ++rounds;
    } else if (cmd != shm::kQuit) {
      h_->status = shm::kBadRequest;
    }
    h_->resp_seq.store(served, std::memory_order_release);
    shm::wake(h_->resp_seq);
    if (cmd == shm::kQuit) return rounds;
  }
}
