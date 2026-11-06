#pragma once
// Host: owns a World on its own thread; the only thread that ever touches the World.
// Clients submit raw dvsim.Envelope bytes from any thread (MPSC queue) and get the dvsim.Reply back.
// The queue is drained only at tick boundaries. The Host starts paused; Run{rate} free-runs one tick at
// a time paced to rate sim-seconds per wall-second (0 = as fast as possible), Pause freezes the tick
// (the thread then blocks on the queue). Run/Pause are logged like any control command, so the log
// (World::log, replay()) reproduces the session, including the ticks advanced between commands.
//
// Viewers read through take_frame(): the sim thread copies the render state (poses, plus the visuals when the scene
// changed) into a back buffer when a frame was requested, the viewer swaps it out. The viewer never touches the World.
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "core/world.h"

// One consistent snapshot of what a viewer draws (all copied by the sim thread at a tick boundary).
struct Frame {
  std::uint64_t tick = 0;
  std::uint64_t scene = 0;           // changes when the model was replaced (LoadModel): re-read prims, re-upload visuals
  bool new_scene = false;            // visuals below are valid (first frame, and after each scene change)
  std::vector<GeomVisual> visuals;
  std::vector<GeomPose> poses;       // index-parallel to visuals
  std::vector<TubeVisual> tubes;     // every frame
  std::vector<std::pair<std::string, std::string>> tube_ends;  // index-parallel to tubes: the instance ids at each cable's ends
  Weather weather;                   // every frame: rain rate etc. for the renderer (all zero = clear)
  double dt = 0;                     // seconds per tick
  bool running = false, diverged = false;
  double rate = 0;                   // Run rate while running (0 = as fast as possible)
};

class Host {
 public:
  // Takes the World. Starts the replay log unless the World already has one or has snapshots (then the Host runs
  // unlogged and world().log() stays null).
  // ponytail: per-submit sessions arrive with the MCP/viewer front ends (M8+); today `elevate` is the only difference.
  Host(World world, Session session);
  ~Host() { stop(); }
  Host(const Host&) = delete;
  Host& operator=(const Host&) = delete;

  // Thread-safe. Throws std::runtime_error after stop(). A job still queued when stop() runs is dropped:
  // its future throws std::future_error (broken_promise). elevate: run this one command with all capabilities
  // (the caller has the approval); the session itself is unchanged.
  std::future<flatbuffers::DetachedBuffer> submit(std::vector<std::uint8_t> envelope, bool elevate = false);
  flatbuffers::DetachedBuffer call(std::vector<std::uint8_t> envelope, bool elevate = false) {
    return submit(std::move(envelope), elevate).get();
  }
  // Runs fn(World&) on the sim thread at a tick boundary and waits (exceptions propagate). For reads and clones
  // that have no bus command; anything that changes state must go through submit() so it is logged.
  void exec(std::function<void(World&)> fn);

  // Viewer side, non-blocking: if the sim thread has produced a frame since the last call, moves it into `out` and
  // returns true. Always asks the sim thread for a fresh one (so call it once per drawn frame).
  bool take_frame(Frame& out);

  void stop();  // idempotent; joins the sim thread
  World& world();  // only after stop() (throws std::logic_error otherwise)

 private:
  struct Job {
    std::vector<std::uint8_t> env;
    bool elevate = false;
    std::function<void(World&)> fn;  // set: run this instead of executing `env`
    std::promise<flatbuffers::DetachedBuffer> reply;
  };
  std::future<flatbuffers::DetachedBuffer> enqueue(Job j);
  void loop();
  Frame snapshot(bool running, double rate);

  World world_;
  Session session_, elevated_;
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Job> q_;
  bool stop_ = false;
  bool want_frame_ = false, have_frame_ = false;  // under m_
  Frame back_;                                    // under m_: written by the sim thread, swapped out by take_frame
  std::uint64_t scene_ = 1;                       // sim thread only
  bool scene_sent_ = false;                       // sim thread only: the current scene's visuals reached a Frame
  std::thread thread_;  // last: starts after everything above is constructed
};
