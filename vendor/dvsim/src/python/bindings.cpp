// _dvsim: thin nanobind layer over World. All logic stays in C++; python/dvsim/__init__.py adds only
// dict<->JSON and numpy conveniences. Every call that can run long releases the GIL; a World is
// single-threaded, so one Sim must not be used from two threads at once (different Sims are independent).
//
// Two backends expose the same methods (bind_common): `Sim` calls its World directly on the caller's thread;
// `HostSim` owns a Host (the World lives on the Host's sim thread, so a viewer can run beside it) and routes every call
// through it. Python's dvsim.Sim wraps either one.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "core/host.h"
#include "core/world.h"
#include "lodepng.h"
#include "mujoco/mujoco_solver.h"
#include "dvsim_generated.h"
#include "dvsim_schema_json.h"
#include "render/renderer.h"
#ifdef DVSIM_HAS_VIEWER
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include "viewer/hud.h"
#include "viewer/viewer.h"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace {
World make_world(const std::string& path) {
  World w(load_mujoco, path);
  w.set_render_hook(make_render_hook());  // lazy: Vulkan is only touched by the first Render
  return w;
}
Session make_session(const std::string& path, const std::string& root, int max_sweep_runs, double max_wall_s,
                     std::vector<std::string> caps, int max_steps_per_call) {
  Session s;
  s.caps = std::move(caps);  // plain Python: all caps (trusted caller); the MCP layer passes read+control
  s.max_steps_per_call = max_steps_per_call;  // plain Python: INT_MAX; the MCP layer passes a finite limit
  s.max_sweep_runs = max_sweep_runs;  // plain Python trusts its caller: generous defaults (see Sim.load)
  s.max_wall_s = max_wall_s;
  s.root = root.empty() ? std::filesystem::path(path).parent_path() : std::filesystem::path(root);
  return s;
}
Session elevated(Session s) {
  s.caps = {"read", "control", "destructive"};  // one call, after the caller's approval
  return s;
}

struct Sim {
  World w;
  Session s;

  Sim(const std::string& path, const std::string& root, int max_sweep_runs, double max_wall_s,
      std::vector<std::string> caps, int max_steps_per_call)
      : w(make_world(path)), s(make_session(path, root, max_sweep_runs, max_wall_s, std::move(caps), max_steps_per_call)) {}
  Sim(World&& world, Session session) : w(std::move(world)), s(std::move(session)) {}

  flatbuffers::DetachedBuffer exec(std::span<const std::uint8_t> env, bool elevate) {
    return w.execute(env, elevate ? elevated(s) : s);
  }
  template <class F>
  void with_world(F&& f) {
    f(w);
  }
};

struct HostSim {
  Session s;  // the Host's session (caps are reported from here)
  std::unique_ptr<Host> h;
  std::atomic<bool> close{false};  // request_close(): ends a running view()

  HostSim(const std::string& path, const std::string& root, int max_sweep_runs, double max_wall_s,
          std::vector<std::string> caps, int max_steps_per_call)
      : s(make_session(path, root, max_sweep_runs, max_wall_s, std::move(caps), max_steps_per_call)),
        h(std::make_unique<Host>(make_world(path), s)) {}

  flatbuffers::DetachedBuffer exec(std::span<const std::uint8_t> env, bool elevate) {
    return h->call(std::vector<std::uint8_t>(env.begin(), env.end()), elevate);
  }
  template <class F>
  void with_world(F&& f) {  // f runs on the sim thread: C++ only (no Python objects)
    h->exec([&](World& w) { f(w); });
  }
};

using Array = nb::ndarray<nb::numpy, double>;

// numpy array owning a copy of [src, src+n).
template <class T>
nb::ndarray<nb::numpy, T> copy_array(const T* src, std::initializer_list<std::size_t> shape) {
  std::size_t n = 1;
  for (std::size_t d : shape) n *= d;
  auto* data = new T[n];
  std::copy(src, src + n, data);
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::ndarray<nb::numpy, T>(data, shape, owner);
}

// Runs one command through the bus with the GIL released and returns the binary Reply.
template <class T>
flatbuffers::DetachedBuffer run_envelope(T& s, flatbuffers::FlatBufferBuilder& fbb, bool elevate = false) {
  nb::gil_scoped_release nogil;
  return s.exec({fbb.GetBufferPointer(), fbb.GetSize()}, elevate);
}

// Runs one read command through the bus (capability-checked, binary reply) and hands the Reply to `use`, which
// copies out what it needs while the buffer is alive. A refused command becomes ValueError (invalid_argument) or
// RuntimeError (everything else, message prefixed with the error name).
template <class T, class F>
auto read_command(T& s, flatbuffers::FlatBufferBuilder& fbb, F use) {
  const flatbuffers::DetachedBuffer reply = run_envelope(s, fbb);
  const dvsim::Reply* r = flatbuffers::GetRoot<dvsim::Reply>(reply.data());
  if (!r->ok()) {
    const std::string msg = std::string(dvsim::EnumNameErr(r->error())) + ": " + (r->message() ? r->message()->str() : "");
    if (r->error() == dvsim::Err::invalid_argument) throw nb::value_error(msg.c_str());
    throw std::runtime_error(msg);
  }
  return use(*r);
}

template <class T>
void bind_common(nb::class_<T>& c) {
  c  // Step goes through the bus (not World::step) so it is logged and replayable.
      .def("step",
           [](T& s, int n) {
             flatbuffers::FlatBufferBuilder fbb;
             fbb.Finish(dvsim::CreateEnvelope(fbb, 0, fbb.CreateString("python"), dvsim::Command::Step,
                                             dvsim::CreateStep(fbb, n).Union()));
             const flatbuffers::DetachedBuffer reply = run_envelope(s, fbb);
             const dvsim::Reply* r = flatbuffers::GetRoot<dvsim::Reply>(reply.data());
             if (!r->ok()) throw std::runtime_error(r->message()->str());
             return r->tick();
           },
           "n"_a = 1)
      // Render / Raycast go through the bus (read capability) but take the binary reply, so pixels, depth and
      // distances reach numpy losslessly (the JSON edge would round doubles and print every pixel).
      .def("render",
           [](T& s, std::vector<double> pos, std::vector<double> lookat, double fovy, int width, int height) {
             flatbuffers::FlatBufferBuilder fbb;
             fbb.Finish(dvsim::CreateEnvelope(fbb, 0, fbb.CreateString("python"), dvsim::Command::Render,
                                             dvsim::CreateRender(fbb, fbb.CreateVector(pos), fbb.CreateVector(lookat), fovy,
                                                                width, height).Union()));
             return read_command(s, fbb, [](const dvsim::Reply& r) {
               const std::size_t w = static_cast<std::size_t>(r.width()), h = static_cast<std::size_t>(r.height());
               return std::make_tuple(copy_array<std::uint8_t>(r.image()->data(), {h, w, 4}),
                                      copy_array<float>(r.depth()->data(), {h, w}));
             });
           },
           "pos"_a, "lookat"_a, "fovy"_a = 45.0, "width"_a = 320, "height"_a = 240)
      .def("raycast",
           [](T& s, std::vector<double> origin, std::vector<double> directions) {
             flatbuffers::FlatBufferBuilder fbb;
             fbb.Finish(dvsim::CreateEnvelope(fbb, 0, fbb.CreateString("python"), dvsim::Command::Raycast,
                                             dvsim::CreateRaycast(fbb, fbb.CreateVector(origin), fbb.CreateVector(directions)).Union()));
             return read_command(s, fbb, [](const dvsim::Reply& r) {
               return copy_array<double>(r.distances()->data(), {r.distances()->size()});
             });
           },
           "origin"_a, "directions"_a)
      // Batch of single-value SetAttrs (SP4 dog leg control, 36-60 per 2 ms tick): one Envelope per entry through the same command bus as
      // call(), so each is logged and replayed exactly like a JSON SetAttr; only the JSON round trip is skipped. Stops at the first failure.
      .def("set_many",
           [](T& s, const std::vector<std::string>& paths, const std::vector<std::string>& attrs, const std::vector<double>& values) {
             if (paths.size() != attrs.size() || paths.size() != values.size()) throw nb::value_error("set_many: lengths differ");
             for (std::size_t i = 0; i < paths.size(); ++i) {
               flatbuffers::FlatBufferBuilder fbb;
               const double v = values[i];
               fbb.Finish(dvsim::CreateEnvelope(fbb, 0, fbb.CreateString("python"), dvsim::Command::SetAttr,
                                               dvsim::CreateSetAttr(fbb, fbb.CreateString(paths[i]), fbb.CreateString(attrs[i]),
                                                                   fbb.CreateString(""), fbb.CreateVector(&v, 1)).Union()));
               read_command(s, fbb, [](const dvsim::Reply&) { return 0; });
             }
           },
           "paths"_a, "attrs"_a, "values"_a)
      // The cables the renderer draws as tubes (spatial tendons through sites): [(radius, length, rgba, points (n, 3))], world frame.
      .def("tubes",
           [](T& s) {
             std::vector<TubeVisual> t;
             s.with_world([&](World& w) { t = w.solver().tubes(); });
             std::vector<std::tuple<double, double, std::vector<float>, nb::ndarray<nb::numpy, double>>> out;
             for (const TubeVisual& v : t)
               out.emplace_back(v.radius, v.length, std::vector<float>(v.rgba, v.rgba + 4), copy_array<double>(v.pts.data(), {v.pts.size() / 3, 3}));
             return out;
           })
      .def_prop_ro("tick", [](T& s) {
        std::uint64_t t = 0;
        s.with_world([&](World& w) { t = w.tick(); });
        return t;
      })
      .def_prop_ro("diverged", [](T& s) {
        bool d = false;
        s.with_world([&](World& w) { d = w.diverged(); });
        return d;
      })
      .def("call",
           [](T& s, const std::string& json, bool elevate) {
             nb::gil_scoped_release nogil;
             return execute_json([&](std::span<const std::uint8_t> env) { return s.exec(env, elevate); },
                                 [&] {
                                   std::uint64_t t = 0;
                                   s.with_world([&](World& w) { t = w.tick(); });
                                   return t;
                                 },
                                 json);
           },
           "json"_a, "elevate"_a = false)
      .def_prop_ro("caps", [](T& s) { return s.s.caps; })
      // Direct solver read (reads are never logged and change no state). Writes must use call().
      .def("get",
           [](T& s, const std::string& path, const std::string& attr, const std::string& frame) {
             std::vector<double> v;
             std::string unit;
             Result r;
             {
               nb::gil_scoped_release nogil;
               s.with_world([&](World& w) {
                 r = w.solver().get(path, attr, frame, v);
                 if (r.code == Result::ok) unit = w.solver().unit(path, attr);
               });
             }
             if (r.code == Result::not_found) throw nb::key_error(r.message.c_str());
             if (r.code != Result::ok) throw nb::value_error(r.message.c_str());
             auto* data = new double[v.size()];
             std::copy(v.begin(), v.end(), data);
             nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
             return std::make_tuple(Array(data, {v.size()}, owner), unit);
           },
           "path"_a, "attr"_a, "frame"_a = "")
      .def("save", [](T& s) {
        std::vector<std::byte> b;
        {
          nb::gil_scoped_release nogil;
          s.with_world([&](World& w) { b = w.save(); });
        }
        return nb::bytes(reinterpret_cast<const char*>(b.data()), b.size());
      })
      .def("restore",
           [](T& s, nb::bytes b) {
             const std::vector<std::byte> bytes(reinterpret_cast<const std::byte*>(b.c_str()),
                                                reinterpret_cast<const std::byte*>(b.c_str()) + b.size());
             nb::gil_scoped_release nogil;
             s.with_world([&](World& w) { w.restore(bytes); });  // throws on malformed
           },
           "state"_a)
      .def("state_hash", [](T& s) {
        std::uint64_t h = 0;
        s.with_world([&](World& w) { h = w.state_hash(); });
        return h;
      })
      .def("clone", [](T& s) {
        std::optional<Sim> out;
        {
          nb::gil_scoped_release nogil;
          s.with_world([&](World& w) { out.emplace(w.clone(), s.s); });
        }
        return std::move(*out);
      })
      .def("log", [](T& s) -> nb::object {
        std::optional<std::vector<std::byte>> b;
        s.with_world([&](World& w) {
          if (w.log()) b = w.log()->bytes();
        });
        if (!b) return nb::none();
        return nb::bytes(reinterpret_cast<const char*>(b->data()), b->size());
      });
}

#ifdef DVSIM_HAS_VIEWER
// Runs the viewer on this thread with the GIL released. A Python KeyboardInterrupt is not delivered until it returns.
// A Python callable (JSON request -> JSON reply, dvsim.design.viewer.Backend) the viewer calls from its worker threads: each call
// takes the GIL, and so does dropping the last reference (the viewer runs with the GIL released).
struct PyApi {
  std::shared_ptr<nb::object> fn;
  // ttsr-ignore: cpp/no-raw-new, cpp/no-raw-delete (a shared_ptr with a GIL-taking deleter: the object must die under the GIL)
  explicit PyApi(nb::object o)
      : fn(new nb::object(std::move(o)), [](nb::object* p) {
          nb::gil_scoped_acquire g;
          delete p;
        }) {}
  std::string operator()(const std::string& req) const {
    nb::gil_scoped_acquire g;
    try {
      return nb::cast<std::string>((*fn)(req));
    } catch (const nb::python_error& e) {
      std::string m = e.what();
      for (char& c : m)
        if (c == '"' || c == '\\' || static_cast<unsigned char>(c) < 0x20) c = ' ';
      return "{\"ok\":false,\"error\":\"" + m + "\"}";
    }
  }
};

void view(Host& host, int frames, const std::string& script, int width, int height, const std::atomic<bool>* stop,
          nb::object design, nb::object hud, bool clean) {
  ViewerOptions o;
  o.frames = frames;
  o.script = script;
  o.width = width;
  o.height = height;
  o.stop = stop;
  o.clean = clean;
  if (!design.is_none()) o.design_api = PyApi(std::move(design));
  if (!hud.is_none()) {  // a dvsim.Hud (the caller's reference keeps it alive for the whole call)
    hud::Overlay* h = nb::cast<hud::Overlay*>(hud);
    o.overlay = h->callback();
    o.camera = h->camera_driver();
    o.on_close = [h] { h->release(); };
  }
  nb::gil_scoped_release nogil;
  run_viewer(host, o);  // throws std::runtime_error (-> RuntimeError) with the reason: no display, no Vulkan device, ...
}
#endif
[[noreturn]] void no_viewer() { throw std::runtime_error("this build of dvsim has no viewer (DVSIM_VIEWER=OFF, or a platform without one)"); }
}  // namespace

NB_MODULE(_dvsim, m) {
  install_mujoco_handlers();  // before any thread exists

  m.def("command_cap", [](const std::string& command) { return command_cap(command); }, "command"_a);
  // Process-wide look of Sim.render / the viewer: "studio,edges,noshadows,nowires" (words add to the current style); returns it.
  m.def("render_style",
        [](const std::string& words) {
          const RenderStyle s = parse_render_style(words, default_render_style());
          set_default_render_style(s);
          return std::string(s.sky == RenderStyle::Sky::studio ? "studio" : "outdoor") + (s.edges ? ",edges" : "") +
                 (s.shadows ? "" : ",noshadows") + (s.tubes ? "" : ",nowires") + (s.wire_px > 0 ? ",boldwires" : "") +
                 (s.rain >= 0 ? ",rain=" + std::to_string(s.rain) : "") + (s.overcast >= 0 ? ",overcast=" + std::to_string(s.overcast) : "");
        },
        "words"_a = "");
  m.attr("schema_json") = kSchemaJson;  // flatc --jsonschema output, embedded at build time
#ifdef DVSIM_HAS_VIEWER
  m.attr("has_viewer") = true;
  // Off-screen viewer (DVSIM_OFFSCREEN=1): the newest frame as (seq, (h, w, 4) uint8 RGBA), or None before the first one.
  m.def("viewer_frame", []() -> nb::object {
    std::vector<std::uint8_t> px;
    int w = 0, h = 0;
    const std::uint64_t seq = viewer_latest_frame(&px, &w, &h);
    if (!seq || px.empty()) return nb::none();
    return nb::make_tuple(seq, copy_array<std::uint8_t>(px.data(), {static_cast<std::size_t>(h), static_cast<std::size_t>(w), 4}));
  });
#else
  m.attr("has_viewer") = false;
#endif

  // PNG (lodepng, the copy MuJoCo already ships) of an (h, w, 4) uint8 RGBA array.
  m.def("encode_png",
        [](nb::ndarray<const std::uint8_t, nb::shape<-1, -1, 4>, nb::c_contig, nb::device::cpu> img) {
          std::vector<unsigned char> out;
          if (const unsigned err = lodepng::encode(out, img.data(), static_cast<unsigned>(img.shape(1)),
                                                   static_cast<unsigned>(img.shape(0))))
            throw std::runtime_error(std::string("PNG encoding failed: ") + lodepng_error_text(err));
          return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
        },
        "rgba"_a);

  nb::class_<Sim> sim(m, "Sim");
  sim.def("__init__",
          [](Sim* self, const std::string& path, const std::string& root, int max_sweep_runs, double max_wall_s,
             std::vector<std::string> caps, int max_steps_per_call) {
            nb::gil_scoped_release nogil;
            new (self) Sim(path, root, max_sweep_runs, max_wall_s, std::move(caps), max_steps_per_call);  // throws std::runtime_error (-> RuntimeError) with MuJoCo's message
          },
          "path"_a, "root"_a = "", "max_sweep_runs"_a = 100000, "max_wall_s"_a = 0.0,
          "caps"_a = std::vector<std::string>{"read", "control", "destructive"}, "max_steps_per_call"_a = INT_MAX);
  bind_common(sim);
  sim.def("start_log", [](Sim& s) { s.w.start_log(); })
      // Hands the World to a Host (its own sim thread) and runs the viewer on this thread until the window closes or
      // `frames` frames were drawn; then takes the World back. The session's commands are logged like any Host's.
      .def("show",
           [](Sim& s, int frames, const std::string& script, int width, int height, nb::object design, nb::object hud, bool clean) {
#ifdef DVSIM_HAS_VIEWER
             Host host(std::move(s.w), s.s);
             struct TakeBack {  // also on an exception: the Sim must get its World back
               Sim& s;
               Host& h;
               ~TakeBack() {
                 h.stop();
                 s.w = std::move(h.world());
               }
             } take_back{s, host};
             view(host, frames, script, width, height, nullptr, std::move(design), std::move(hud), clean);
#else
             no_viewer();
#endif
           },
           "frames"_a = 0, "script"_a = "", "width"_a = 1280, "height"_a = 720, "design"_a = nb::none(), "hud"_a = nb::none(),
           "clean"_a = false);

  // A Sim whose World lives on a Host thread: what `dvsim mcp --view` serves, with the viewer on the main thread.
  nb::class_<HostSim> host(m, "Host");
  host.def("__init__",
           [](HostSim* self, const std::string& path, const std::string& root, int max_sweep_runs, double max_wall_s,
              std::vector<std::string> caps, int max_steps_per_call) {
             nb::gil_scoped_release nogil;
             new (self) HostSim(path, root, max_sweep_runs, max_wall_s, std::move(caps), max_steps_per_call);
           },
           "path"_a, "root"_a = "", "max_sweep_runs"_a = 100000, "max_wall_s"_a = 0.0,
           "caps"_a = std::vector<std::string>{"read", "control", "destructive"}, "max_steps_per_call"_a = INT_MAX);
  bind_common(host);
  host.def("view",
           [](HostSim& s, int frames, const std::string& script, int width, int height, nb::object design, nb::object hud, bool clean) {
#ifdef DVSIM_HAS_VIEWER
             view(*s.h, frames, script, width, height, &s.close, std::move(design), std::move(hud), clean);  // request_close() latches: later views return at once
#else
             no_viewer();
#endif
           },
           "frames"_a = 0, "script"_a = "", "width"_a = 1280, "height"_a = 720, "design"_a = nb::none(), "hud"_a = nb::none(),
           "clean"_a = false)
      .def("request_close", [](HostSim& s) { s.close = true; })  // callable from any thread
      .def("stop", [](HostSim& s) {
        nb::gil_scoped_release nogil;
        s.h->stop();
      });

#ifdef DVSIM_HAS_VIEWER
  // The viewer's data-driven HUD (src/viewer/hud.h): `Sim.show(hud=Hud())`, then `hud.set(json)` from any thread.
  auto labels_of = [](const std::vector<hud::Label>& v) {
    nb::list out;
    for (const hud::Label& l : v) out.append(nb::make_tuple(l.text, l.x, l.y, l.visible));
    return out;
  };
  nb::class_<hud::Overlay>(m, "Hud")
      .def(nb::init<>())
      .def("set", [](hud::Overlay& h, const std::string& json) { h.set(json); }, "json"_a)  // bad JSON -> RuntimeError
      // -> dict(seq, tick, stable, frames, error): what the window drew last (see hud::Overlay::Progress)
      .def("progress", [](hud::Overlay& h) {
        const hud::Overlay::Progress p = h.progress();
        nb::dict d;
        d["seq"] = p.seq, d["tick"] = p.tick, d["stable"] = p.stable, d["frames"] = p.frames, d["error"] = p.error;
        return d;
      })
      .def("labels", [labels_of](hud::Overlay& h) { return labels_of(h.labels()); })  // [(text, x, y, visible)] of the last frame
      // Headless draw of the current document (no window, no GPU) -> dict(labels, vertices, error). Not while a viewer runs.
      .def("dry_run",
           [labels_of](hud::Overlay& h, std::array<double, 3> pos, std::array<double, 3> look, double fovy, double w, double hgt,
                       std::array<double, 4> vp) {
             OverlayCtx c;
             for (int k = 0; k < 3; ++k) c.cam_pos[k] = pos[k], c.cam_look[k] = look[k];
             c.cam_fovy = fovy, c.window_w = w, c.window_h = hgt;
             c.x = static_cast<float>(vp[0]), c.y = static_cast<float>(vp[1]), c.w = static_cast<float>(vp[2]), c.h = static_cast<float>(vp[3]);
             if (c.w <= 0) c.x = 0, c.y = 0, c.w = static_cast<float>(w), c.h = static_cast<float>(hgt);
             const hud::Overlay::DryRun r = h.dry_run(c);
             nb::dict d;
             d["labels"] = labels_of(r.labels), d["vertices"] = r.vertices, d["error"] = r.error;
             return d;
           },
           "cam_pos"_a, "cam_look"_a, "fovy"_a, "width"_a, "height"_a, "viewport"_a = std::array<double, 4>{0, 0, 0, 0});
#endif
}
