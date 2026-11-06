#include "viewer/viewer.h"

#include <volk.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <imgui_internal.h>  // (the list of live windows, for ui.report)

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "dvsim_generated.h"
#include "render/window_renderer.h"
#include "viewer/build3d.h"
#include "viewer/design.h"
#include "viewer/library_ui.h"
#include "viewer/program_ui.h"
#include "viewer/run_ui.h"
#include "viewer/theme.h"
#include "viewer/wire2d.h"

namespace {
using Env = std::vector<std::uint8_t>;
using Reply = flatbuffers::DetachedBuffer;
constexpr const char* kSource = "viewer";

// ---- command builders (the one place envelopes are made: UI controls and scripts both use them) ------------------------
template <class F>
Env envelope(const std::string& source, dvsim::Command type, F make) {
  static std::uint64_t next_id = 1;  // viewer thread only (the script test helper is single-threaded too)
  flatbuffers::FlatBufferBuilder fbb;
  const auto src = fbb.CreateString(source);
  const auto cmd = make(fbb);
  fbb.Finish(dvsim::CreateEnvelope(fbb, next_id++, src, type, cmd));
  return Env(fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize());
}
Env env_run(const std::string& src, double rate) {
  return envelope(src, dvsim::Command::Run, [&](auto& f) { return dvsim::CreateRun(f, rate).Union(); });
}
Env env_pause(const std::string& src) {
  return envelope(src, dvsim::Command::Pause, [&](auto& f) { return dvsim::CreatePause(f).Union(); });
}
Env env_step(const std::string& src, int n) {
  return envelope(src, dvsim::Command::Step, [&](auto& f) { return dvsim::CreateStep(f, n).Union(); });
}
Env env_setattr(const std::string& src, const std::string& path, const std::string& attr, const std::vector<double>& v) {
  return envelope(src, dvsim::Command::SetAttr,
                  [&](auto& f) { return dvsim::CreateSetAttrDirect(f, path.c_str(), attr.c_str(), "", &v).Union(); });
}
Env env_load(const std::string& src, const std::string& path) {
  return envelope(src, dvsim::Command::LoadModel, [&](auto& f) { return dvsim::CreateLoadModelDirect(f, path.c_str()).Union(); });
}
Env env_query(const std::string& src, const std::string& path, const std::string& attr) {
  return envelope(src, dvsim::Command::Query,
                  [&](auto& f) { return dvsim::CreateQueryDirect(f, path.c_str(), attr.c_str(), "").Union(); });
}

// Design-window script actions (see viewer.h): {min, max} argument counts; {-1, -1} = not one.
std::pair<int, int> design_action_args(const std::string& a) {
  static const std::map<std::string, std::pair<int, int>> t = {
      {"dsuggest", {1, 2}}, {"dsnap", {2, 3}}, {"dcycle", {1, 1}}, {"dswap", {2, 2}}, {"dremove", {1, 1}}, {"dautowire", {1, 2}},
      {"dautowire_all", {0, 0}}, {"dundo", {0, 0}}, {"dredo", {0, 0}}, {"dsave", {0, 1}}, {"dsim", {1, 1}}, {"dfinding", {1, 1}},
      {"dsync", {0, 0}}};
  const auto it = t.find(a);
  return it == t.end() ? std::pair<int, int>{-1, -1} : it->second;
}

// UI-only script actions (window state, no bus command, no editor needed): {min, max} argument counts; {-1, -1} = not one.
std::pair<int, int> ui_action_args(const std::string& a) {
  static const std::map<std::string, std::pair<int, int>> t = {
      {"mode", {1, 1}}, {"uisel", {1, 1}}, {"search", {1, 12}}, {"uicam", {1, 1}}, {"uiframe", {0, 0}}, {"uikeys", {0, 1}}, {"crumb", {1, 1}}};
  const auto it = t.find(a);
  return it == t.end() ? std::pair<int, int>{-1, -1} : it->second;
}

double to_double(const std::string& s, const std::string& what) {
  std::size_t used = 0;
  double v = 0;
  try {
    v = std::stod(s, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used != s.size() || s.empty()) throw std::runtime_error("script: " + what + " must be a number, got '" + s + "'");
  return v;
}

// ---- GLFW / ImGui lifetime (destroyed in reverse order of declaration) ---------------------------------------------------
std::string g_glfw_error;
struct GlfwInit {
  GlfwInit() {
    glfwSetErrorCallback([](int, const char* d) { g_glfw_error = d ? d : ""; });
#if defined(__linux__)
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);  // the only backend built here; gives the specific "no DISPLAY" message
#endif
    if (!glfwInit()) throw std::runtime_error("cannot open a window: " + (g_glfw_error.empty() ? "GLFW init failed" : g_glfw_error));
  }
  ~GlfwInit() { glfwTerminate(); }
};
struct GlfwWindow {
  GLFWwindow* w;
  GlfwWindow(int width, int height) : w(glfwCreateWindow(width, height, "dvsim", nullptr, nullptr)) {
    if (!w) throw std::runtime_error("cannot open a window: " + (g_glfw_error.empty() ? "glfwCreateWindow failed" : g_glfw_error));
  }
  ~GlfwWindow() { glfwDestroyWindow(w); }
};
void vk_check(VkResult r) {
  if (r != VK_SUCCESS) throw std::runtime_error("ImGui Vulkan backend error (VkResult " + std::to_string(r) + ")");
}
struct ImGuiBackends {
  ImGuiBackends(GLFWwindow* win, const WindowRenderer::Handles& h) : fmt(h.color_format), glfw(win != nullptr) {  // win null: off-screen, no platform backend
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini in the working directory
    if (win) ImGui_ImplGlfw_InitForVulkan(win, true);
    ImGui_ImplVulkan_InitInfo ii = {};
    ii.ApiVersion = VK_API_VERSION_1_3;
    ii.Instance = h.instance;
    ii.PhysicalDevice = h.physical_device;
    ii.Device = h.device;
    ii.QueueFamily = h.queue_family;
    ii.Queue = h.queue;
    // One sampled-image descriptor per font atlas plus one per registered texture (library thumbnails, HUD images). radv
    // enforces the pool size (VK_ERROR_FRAGMENTED_POOL on the 9th texture); llvmpipe never did.
    ii.DescriptorPoolSize = 4096;
    ii.MinImageCount = 2;
    ii.ImageCount = 3;
    ii.UseDynamicRendering = true;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &fmt;  // fmt outlives the backend
    ii.CheckVkResultFn = vk_check;
    if (!ImGui_ImplVulkan_Init(&ii)) throw std::runtime_error("ImGui Vulkan backend failed to initialise");
  }
  ~ImGuiBackends() {
    ImGui_ImplVulkan_Shutdown();
    if (glfw) ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
  }
  VkFormat fmt;
  bool glfw;
};

// The newest off-screen frame (see viewer_latest_frame in viewer.h).
std::mutex g_frame_mu;
std::vector<std::uint8_t> g_frame;
int g_frame_w = 0, g_frame_h = 0;
std::uint64_t g_frame_seq = 0;

// ---- scene tree --------------------------------------------------------------------------------------------------------
struct Node {
  std::string path;
  bool prim = false;
  std::map<std::string, Node> kids;
};
void add_path(Node& root, const std::string& path) {
  Node* n = &root;
  std::string acc;
  for (std::size_t i = path.find_first_not_of('/'); i != std::string::npos;) {
    const std::size_t j = std::min(path.find('/', i), path.size());
    const std::string seg = path.substr(i, j - i);
    acc += "/" + seg;
    n = &n->kids[seg];
    n->path = acc;
    i = j < path.size() ? j + 1 : std::string::npos;
  }
  n->prim = true;
}

// Attributes the inspector asks for; a prim answers only the ones it has (the rest come back not_found / invalid).
constexpr const char* kAttrs[] = {"pos", "quat", "qpos", "qvel", "ctrl", "kp", "kv", "value"};
bool editable(const std::string& a) { return a == "qpos" || a == "qvel" || a == "ctrl" || a == "kp" || a == "kv"; }

struct Shown {
  std::string attr, unit;
  std::vector<double> values;
};

// Orbit camera around `target` (world frame, +z up).
struct Orbit {
  double yaw = -0.6, pitch = 0.35, dist = 4, target[3] = {0, 0, 0};
  Camera camera() const {
    Camera c;
    c.lookat[0] = target[0];
    c.lookat[1] = target[1];
    c.lookat[2] = target[2];
    c.pos[0] = target[0] + dist * std::cos(pitch) * std::cos(yaw);
    c.pos[1] = target[1] + dist * std::cos(pitch) * std::sin(yaw);
    c.pos[2] = target[2] + dist * std::sin(pitch);
    return c;
  }
  // Frame the movable geoms (planes are unbounded floors, terrain is a floor too: ignored).
  void fit(const std::vector<GeomVisual>& vis, const std::vector<GeomPose>& poses) {
    double sum[3] = {0, 0, 0};
    int n = 0;
    for (std::size_t i = 0; i < vis.size() && i < poses.size(); ++i)
      if (vis[i].type != GeomType::plane && vis[i].type != GeomType::hfield && vis[i].group != 3) {
        for (int k = 0; k < 3; ++k) sum[k] += poses[i].pos[k];
        ++n;
      }
    double radius = 1;
    if (n) {
      for (int k = 0; k < 3; ++k) target[k] = sum[k] / n;
      for (std::size_t i = 0; i < vis.size() && i < poses.size(); ++i)
        if (vis[i].type != GeomType::plane && vis[i].type != GeomType::hfield && vis[i].group != 3) {
          double d2 = 0;
          for (int k = 0; k < 3; ++k) d2 += (poses[i].pos[k] - target[k]) * (poses[i].pos[k] - target[k]);
          radius = std::max(radius, std::sqrt(d2) + std::max({vis[i].size[0], vis[i].size[1], vis[i].size[2]}));
        }
    }
    dist = 2.5 * radius;
  }
};

class Viewer {
 public:
  Viewer(Host& h, const ViewerOptions& o) : host_(h), opts_(o), script_(parse_script(o.script)) {
    if (!o.design_report.empty()) {
      design_ = design::load_report(o.design_report);  // throws a plain-language error before any window opens
      has_design_ = true;
    }
    for (const ScriptCmd& c : script_)
      if (design_action_args(c.action).first >= 0 && !o.design_api)
        throw std::runtime_error("script: " + c.action + " needs a design editor (Sim.show(design=...) / dvsim.design.viewer.view)");
    prog_.send = [this](const std::string& kind, const std::string& json) { dreq(kind, json, true); };  // prog.* / term.* ride the design queue
    run_.send = [this](const std::string& kind, const std::string& json) { dreq(kind, json, true); };  // so do run.*
    run_.selected = [this] { const design::Instance* i = sel_instance(); return i ? i->id : std::string(); };
    prog_.run_slot = [this](ImVec2 size) { run_.draw(size); };
  }

  void run() {
    // Off-screen (opts_.offscreen or DVSIM_OFFSCREEN=1): no GLFW, no window, no swapchain, no X server; every frame is read
    // back to RGBA (viewer_latest_frame). ImGui gets its display size / delta time from here instead of the GLFW backend.
    const char* off_env = std::getenv("DVSIM_OFFSCREEN");
    const bool off = opts_.offscreen || (off_env && *off_env && std::string(off_env) != "0");
    std::optional<GlfwInit> glfw;
    std::optional<GlfwWindow> window;
    WindowRenderer::Surface surf;
    if (off) {
      surf.offscreen = true;
    } else {
      glfw.emplace();
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      window.emplace(opts_.width, opts_.height);
      if (!glfwVulkanSupported()) throw std::runtime_error("no Vulkan loader found (GLFW could not load libvulkan)");
      std::uint32_t n = 0;
      const char** exts = glfwGetRequiredInstanceExtensions(&n);
      if (!exts) throw std::runtime_error("this window system cannot present Vulkan images (" + g_glfw_error + ")");
      surf.instance_extensions.assign(exts, exts + n);
      surf.create = [&](VkInstance inst) {
        VkSurfaceKHR s = nullptr;
        const VkResult r = glfwCreateWindowSurface(inst, window->w, nullptr, &s);
        if (r != VK_SUCCESS)
          throw std::runtime_error("cannot create a Vulkan surface for the window (VkResult " + std::to_string(r) + "): " + g_glfw_error);
        return s;
      };
    }
    WindowRenderer renderer(surf);
    if (off) std::fprintf(stderr, "dvsim-view: off-screen %dx%d on %s\n", opts_.width, opts_.height, renderer.device_name().c_str());
    if (opts_.design_api) {  // the build view looks like a studio; DVSIM_RENDER_STYLE still overrides
      RenderStyle st = default_render_style();
      st.sky = RenderStyle::Sky::studio;
      set_default_render_style(st);
    }
    ImGuiBackends gui(window ? window->w : nullptr, renderer.handles());
    theme::init();
    if (opts_.clean) {  // a recording / HUD view: no pointer drawn over the picture
      if (window) glfwSetInputMode(window->w, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
      ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
    }
    libui::Thumbs thumbs;
    thumbs_ = &thumbs;

    if (opts_.design_api) {
      d_state(true);  // the design behind the model on screen (quiet: an MCP server may not have one yet)
      dreq("cats", design::request("library.categories"), true);
      dreq("mypart", design::request("library.my_parts"), true);  // MY PARTS count
      dreq("search", [this] { return search_req(); }, true);  // the first page of the library (an empty query lists everything)
    }
    using Clock = std::chrono::steady_clock;
    auto last = Clock::now();
    std::vector<std::uint8_t> shot;
    for (int n_frame = 0; !(window && glfwWindowShouldClose(window->w)) && !(opts_.frames > 0 && n_frame >= opts_.frames) &&
                          !(opts_.stop && opts_.stop->load());
         ++n_frame) {
      if (window) glfwPollEvents();
      for (const ScriptCmd& c : script_) {
        if (c.frame != n_frame) continue;
        if (c.action == "select") select(c.args[0]);
        else if (c.action == "inst") select_instance(c.args[0]);
        else if (ui_action_args(c.action).first >= 0) script_ui(c);
        else if (design_action_args(c.action).first >= 0) script_design(c);
        else send(script_envelope(c, kSource), c.action);
      }
      Frame f;
      if (host_.take_frame(f)) adopt(f, renderer);
      poll();

      ImGui_ImplVulkan_NewFrame();
      if (window) {
        ImGui_ImplGlfw_NewFrame();
      } else {
        ImGuiIO& io = ImGui::GetIO();
        const auto now = Clock::now();
        io.DisplaySize = ImVec2(static_cast<float>(opts_.width), static_cast<float>(opts_.height));
        io.DisplayFramebufferScale = ImVec2(1, 1);
        io.DeltaTime = std::max(1e-4f, std::chrono::duration<float>(now - last).count());
        last = now;
        // The pointer sits where Xvfb parks it (the screen centre): popups that open at the pointer land where they did there.
        if (n_frame == 0) io.AddMousePosEvent(opts_.width / 2.f, opts_.height / 2.f);
      }
      ImGui::NewFrame();
      ui();
      ImGui::Render();

      int fw = opts_.width, fh = opts_.height;
      if (window) glfwGetFramebufferSize(window->w, &fw, &fh);
      renderer.set_tubes(shown_tubes());
      renderer.set_weather(cur_.weather);
      const bool drawn = renderer.frame(shown_poses(), view_cam_, fw, fh, [](VkCommandBuffer cb) {
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
      }, window ? nullptr : &shot);
      if (!window) {
        {
          std::lock_guard<std::mutex> lk(g_frame_mu);
          g_frame.swap(shot);
          g_frame_w = fw;
          g_frame_h = fh;
          ++g_frame_seq;
        }
        // ponytail: fixed 60 fps cap (what FIFO vsync gives a window); no free-running GPU loop. Make it an option if a tool wants more.
        std::this_thread::sleep_until(last + std::chrono::microseconds(16667));
      } else if (!drawn) {
        glfwWaitEventsTimeout(0.05);  // minimised or swapchain rebuilding: do not spin
      }
    }
    prog_.shutdown();  // the PROGRAM page's shells die with the window
    draining_ = true;  // edits still queued or running finish (and their model reloads are issued) before we return
    while (dfut_.valid() || !dq_.empty()) {
      poll();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    for (auto& p : cmds_) p.second.wait();  // scripted / UI commands issued last are applied before we return
    poll();
    if (opts_.on_close) opts_.on_close();
    thumbs_ = nullptr;  // (the textures are unregistered by `thumbs` before the ImGui context goes)
  }

 private:
  void send(Env e, const std::string& what, bool elevate = false) { cmds_.emplace_back(what, host_.submit(std::move(e), elevate)); }

  template <class T>
  static bool ready(std::future<T>& f) { return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

  void request_prims() { prims_ = host_.submit(env_query(kSource, "", "")); }

  void adopt(Frame& f, WindowRenderer& renderer) {
    if (f.new_scene) {
      renderer.set_scene(f.visuals);
      if (!keep_camera_) orbit_.fit(f.visuals, f.poses);  // a Design-window rebuild keeps the camera where the person put it
    }
    if (f.scene != frame_scene_) {  // first frame, or the model was replaced: the prim list changed too
      const design::Instance* si = keep_camera_ ? sel_instance() : nullptr;
      const std::string keep = si ? si->id : "";
      if (frame_scene_ != 0) select("");  // (a first-frame selection, e.g. scripted, is kept)
      frame_scene_ = f.scene;
      request_prims();
      keep_camera_ = false;
      if (!keep.empty()) pending_body_ = keep;  // the selected instance stays selected once the new prim list arrives
    }
    f.visuals.clear();
    cur_ = std::move(f);
  }

  // Collects finished replies without blocking the frame.
  void poll() {
    design_poll();
    for (auto it = cmds_.begin(); it != cmds_.end();) {
      if (!ready(it->second)) {
        ++it;
        continue;
      }
      const Reply r = it->second.get();
      const dvsim::Reply* rep = flatbuffers::GetRoot<dvsim::Reply>(r.data());
      status_ = rep->ok() ? "" : it->first + ": " + dvsim::EnumNameErr(rep->error()) + ": " + (rep->message() ? rep->message()->str() : "");
      it = cmds_.erase(it);
    }
    if (prims_.valid() && ready(prims_)) {
      const Reply r = prims_.get();
      tree_ = Node{};
      prim_paths_.clear();
      if (const auto* p = flatbuffers::GetRoot<dvsim::Reply>(r.data())->prims())
        for (const auto* s : *p) {
          add_path(tree_, s->str());
          prim_paths_.push_back(s->str());
        }
      if (!pending_body_.empty()) select_instance(pending_body_);
    }
    if (!inspect_.empty() && std::all_of(inspect_.begin(), inspect_.end(), [](auto& p) { return ready(p.second); })) {
      shown_.clear();
      for (auto& [attr, fut] : inspect_) {
        const Reply r = fut.get();
        const dvsim::Reply* rep = flatbuffers::GetRoot<dvsim::Reply>(r.data());
        if (!rep->ok() || !rep->values()) continue;  // this prim has no such attribute
        shown_.push_back({attr, rep->unit() ? rep->unit()->str() : "", {rep->values()->begin(), rep->values()->end()}});
      }
      inspect_.clear();
    }
    if (inspect_.empty() && !selected_.empty())
      for (const char* a : kAttrs) inspect_.emplace_back(a, host_.submit(env_query(kSource, selected_, a)));
  }

  void select(const std::string& path) {
    selected_ = path;
    shown_.clear();
    inspect_.clear();
  }

  // A Design-window click: selects the body prim of design instance `inst` (by id or body name). Before the prim list
  // has arrived the request is parked and retried when it does.
  void select_instance(const std::string& inst) {
    std::string body = inst;
    for (const design::Instance& i : design_.instances)
      if (i.id == inst) body = i.body;
    const std::string p = design::body_path(prim_paths_, body);
    pending_body_ = p.empty() && prim_paths_.empty() ? inst : "";
    if (!p.empty()) select(p);
  }

  // The design instance whose body prim is selected (nullptr: none / a prim that is not a design instance).
  const design::Instance* sel_instance() const {
    if (!sel_.empty())
      for (const design::Instance& i : design_.instances)
        if (i.id == sel_.back()) return &i;
    for (const design::Instance& i : design_.instances)
      if (selected_.size() > i.body.size() && selected_.ends_with("/" + i.body)) return &i;
    return nullptr;
  }
  // Selects design instances (the last is the primary one: gizmo, inspector); the prim inspector follows the primary.
  void set_sel(std::vector<std::string> ids) {
    std::erase_if(ids, [&](const std::string& i) { return std::none_of(design_.instances.begin(), design_.instances.end(), [&](const design::Instance& x) { return x.id == i; }); });
    if (!ids.empty()) select_instance(ids.back());
    else select("");
    sel_ = std::move(ids);
    if (opts_.design_api && sel_synced_ != sel_) {  // the backend's idea of the selection (agents read it with ui.state)
      sel_synced_ = sel_;
      dreq("quiet", design::request("select", "{\"silent\":true,\"instances\":" + json_list(sel_) + "}"), true);
    }
  }
  bool is_selected(const std::string& id) const { return std::find(sel_.begin(), sel_.end(), id) != sel_.end(); }
  static std::string json_list(const std::vector<std::string>& v) {
    std::string o = "[";
    for (const std::string& s : v) o += (o.size() > 1 ? "," : "") + design::quote(s);
    return o + "]";
  }

  // ---- Design editing: the viewer builds requests and shows replies; dvsim.design.Design (via opts_.design_api) does the work ----
  // One request at a time on a worker thread; a queue keeps UI clicks and scripted actions in order. `make` builds the request
  // when it reaches the head of the queue (so "cycle placement" sees the placement of the snap queued before it); an empty
  // request is a local action only.
  struct DReq {
    std::string kind;
    bool quiet = false;                   // a failure is not shown (polls, the first state)
    std::function<std::string()> make;
  };
  struct Alt {
    std::string a, b;  // port on the target instance / port on the new part
    int k = 0;
    double score = 0;
    bool interferes = false;
    std::vector<std::string> why;
  };
  struct Placement {
    std::string part, id, target;  // target = instance the part is snapped to
    std::vector<Alt> alts;         // [0] = the placement the snap chose, then the ranked alternatives
    int cur = 0;
  };
  struct Sugg {
    std::string part, port, category, name;
    double score = 0, mass = 0, mass_delta = 0;
    int kept = 0, dangling = 0;
    std::vector<std::string> why, lost;
  };
  struct Decision {  // an ambiguous autowire: the user decides which port goes to which candidate
    std::vector<std::string> ports, cands;
    std::string why;
  };

  static std::vector<std::string> strs(const design::Json& j) {
    std::vector<std::string> v;
    for (const design::Json& e : j.arr) v.push_back(e.kind == design::Json::Str ? e.str : "");
    return v;
  }
  static std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string o;
    for (const std::string& s : v) o += (o.empty() ? "" : sep) + s;
    return o;
  }
  static std::string key(const std::string& code, const std::vector<std::string>& where) { return code + "|" + join(where, ","); }
  static std::string q(const std::string& s) { return design::quote(s); }

  void dreq(const std::string& kind, std::function<std::string()> make, bool quiet = false) {
    dq_.push_back({kind, quiet, std::move(make)});
  }
  void dreq(const std::string& kind, std::string json, bool quiet = false) {
    dreq(kind, [j = std::move(json)] { return j; }, quiet);
  }
  static std::string edit_req(const std::string& op, bool replace_last = false) {
    return design::request("design_edit", "{\"ops\":[" + op + "]}", replace_last);
  }

  void d_state(bool quiet = false) { dreq("state", design::request("state"), quiet); }
  void d_suggest(const std::string& target, const std::string& category) {
    sugg_target_ = target;
    sugg_swap_ = target.find('.') == std::string::npos;
    suggs_.clear();
    dreq("suggest", design::request("design_suggest", "{\"target\":" + q(target) + ",\"limit\":12" +
                                                         (category.empty() ? "" : ",\"category\":" + q(category)) + "}"));
  }
  // "snap here": add `part` mated to the best open port pair of `to` ('inst' or 'inst.port'); child_port pins the new part's port.
  void d_snap(const std::string& part, const std::string& to, const std::string& id = "", const std::string& child_port = "") {
    place_ = Placement{part, id, to.substr(0, to.find('.')), {}, 0};
    snap_requested_ = true;
    dreq("edit", edit_req("{\"op\":\"add\",\"part\":" + q(part) + ",\"to\":" + q(to) + (id.empty() ? "" : ",\"id\":" + q(id)) +
                          (child_port.empty() ? "" : ",\"port\":" + q(child_port)) + "}"));
  }
  // Preview placement alternative n of the last snap: undo it and add again with that port pair and rotation (one undo step).
  void d_cycle(int n) {
    dreq("edit", [this, n]() -> std::string {
      if (n < 0 || n >= static_cast<int>(place_.alts.size())) {
        dstatus_ = "cycle: no placement alternative " + std::to_string(n);
        return "";
      }
      const Alt& a = place_.alts[n];
      cycling_ = n;
      return edit_req("{\"op\":\"add\",\"part\":" + q(place_.part) + ",\"id\":" + q(place_.id) + ",\"to\":" + q(place_.target + "." + a.a) +
                          ",\"port\":" + q(a.b) + ",\"k\":" + std::to_string(a.k) + "}", true);
    });
  }
  void d_swap(const std::string& id, const std::string& part, bool auto_dev) {
    dreq("edit", edit_req("{\"op\":\"swap\",\"id\":" + q(id) + ",\"part\":" + q(part) + ",\"auto\":" + (auto_dev ? "true" : "false") + "}"));
  }
  void d_remove(const std::string& id) { dreq("edit", edit_req("{\"op\":\"remove\",\"id\":" + q(id) + "}")); }
  void d_autowire(const std::string& port, const std::string& to = "") {
    dreq("edit", edit_req("{\"op\":\"autowire\",\"a\":" + q(port) + (to.empty() ? "" : ",\"b\":" + q(to)) + "}"));
  }
  void d_autowire_all() { dreq("edit", edit_req("{\"op\":\"autowire_all\"}")); }
  void d_undo(bool redo) { dreq("edit", design::request("design_undo", redo ? "{\"redo\":true}" : "{}")); }
  void d_save(const std::string& path) { dreq("save", design::request("design_save", path.empty() ? "{}" : "{\"path\":" + q(path) + "}")); }
  void d_simulate(const std::string& kind) {
    sim_rows_.clear();
    sim_title_ = kind + ": running...";
    dreq("sim", design::request("design_simulate", "{\"kind\":" + q(kind) + "}"));
  }
  // Click finding n of the DRC list: select the instances it involves (the first in the inspector, all marked in the lists).
  void d_finding(int n) {
    dreq("local", [this, n]() -> std::string {
      if (n < 0 || n >= static_cast<int>(design_.findings.size())) {
        dstatus_ = "finding " + std::to_string(n) + ": no such finding";
        return "";
      }
      finding_ = n;
      hl_.clear();
      const design::Finding& f = design_.findings[n];
      for (const std::string& i : f.instances) hl_.insert(i);
      if (!f.instances.empty()) set_sel(f.instances);
      return "";
    });
  }
  void script_design(const ScriptCmd& c) {
    const auto& a = c.args;
    const std::string arg1 = a.size() > 1 ? a[1] : "";
    if (c.action == "dsuggest") d_suggest(a[0], arg1);
    else if (c.action == "dsnap") d_snap(a[0], a[1], a.size() > 2 ? a[2] : "");
    else if (c.action == "dcycle") d_cycle(static_cast<int>(to_double(a[0], "N")));
    else if (c.action == "dswap") d_swap(a[0], a[1], true);
    else if (c.action == "dremove") d_remove(a[0]);
    else if (c.action == "dautowire") d_autowire(a[0], arg1);
    else if (c.action == "dautowire_all") d_autowire_all();
    else if (c.action == "dundo") d_undo(false);
    else if (c.action == "dredo") d_undo(true);
    else if (c.action == "dsave") d_save(a.empty() ? "" : a[0]);
    else if (c.action == "dsim") d_simulate(a[0]);
    else if (c.action == "dfinding") d_finding(static_cast<int>(to_double(a[0], "N")));
    else if (c.action == "dsync") d_state();
  }

  // Runs on the viewer thread each frame: collects the reply of the request in flight, starts the next one, polls for changes
  // another client (an MCP agent sharing the design) made.
  void design_poll() {
    if (!opts_.design_api) return;
    for (;;) {
      if (dfut_.valid()) {
        if (!ready(dfut_)) return;
        const std::string reply = dfut_.get();
        design_done(dcur_, reply);
      }
      if (dq_.empty()) {
        const auto now = std::chrono::steady_clock::now();
        if (!draining_ && now - last_ev_ > std::chrono::milliseconds(100)) {  // UI events scripted agents queue (ui.mode, drags ...)
          last_ev_ = now;
          dq_.push_back({"uievents", true, [] { return design::request("ui_poll"); }});
        } else if (!draining_ && now - last_poll_ > std::chrono::milliseconds(300)) {
          last_poll_ = now;
          dq_.push_back({"rev", true, [] { return design::request("rev"); }});
        } else if (std::string pk, pj; !draining_ && prog_.poll(ImGui::GetTime(), mode_ == Mode::Program, &pk, &pj)) {
          dq_.push_back({pk, true, [pj] { return pj; }});  // terminal reads: one at a time, only when the queue is idle
        } else if (std::string rk, rj; !draining_ && run_.poll(ImGui::GetTime(), mode_ == Mode::Program, &rk, &rj)) {
          dq_.push_back({rk, true, [rj] { return rj; }});  // run.targets / run.poll (~200 ms) likewise
        } else {
          return;
        }
      }
      dcur_ = std::move(dq_.front());
      dq_.pop_front();
      const std::string json = dcur_.make();
      if (json.empty()) continue;  // a local action (or one that could not be built: dstatus_ says why)
      dfut_ = std::async(std::launch::async, [api = opts_.design_api, json] {
        try {
          return api(json);
        } catch (const std::exception& e) {
          return std::string("{\"ok\":false,\"error\":") + design::quote(e.what()) + "}";
        }
      });
    }
  }

  void design_done(const DReq& rq, const std::string& text) {
    if (rq.kind == "search") lib_busy_ = false;  // answered (or failed): the next dirty query may go out
    design::Json j;
    try {
      j = design::Parser(text).parse();
    } catch (const std::exception& e) {
      dstatus_ = rq.kind + ": " + e.what();
      return;
    }
    if (rq.kind.rfind("prog.", 0) == 0 || rq.kind.rfind("term.", 0) == 0) {  // the PROGRAM page (program_ui.cpp) owns these replies
      prog_.on_reply(rq.kind, j);
      return;
    }
    if (rq.kind.rfind("run.", 0) == 0) {  // the Run slot (run_ui.cpp)
      run_.on_reply(rq.kind, j);
      return;
    }
    if (rq.kind.rfind("flatten.", 0) == 0) {  // the 2D wiring canvas (wire2d.cpp) owns these replies
      const bool was_flat = flat_.active();
      flat_.on_reply(rq.kind, j);
      if (flat_.active() != was_flat) set_mode(flat_.active() ? "wiring" : "build");  // flatten.begin / .end (UI or agent)
      if (j.at("ok").num != 0 && j.at("state").kind == design::Json::Obj) apply_state(j.at("state"));
      return;
    }
    if (j.at("ok").num == 0) {
      if (!rq.quiet) dstatus_ = rq.kind + ": " + j.s("error");
      if (rq.kind == "edit") cycling_ = -1;
      if (rq.kind == "sub") close_failed_ = true;  // (ends a series of sub.close: do_close_docs)
      return;
    }
    dstatus_.clear();
    const design::Json& res = j.at("result");
    if (rq.kind == "rev") {
      flat_.poll(static_cast<int>(res.n("flat_rev")));
      if (const int tr = static_cast<int>(res.n("thumb_rev")); tr != thumb_rev_) {  // library thumbnails landed: ask the library again
        thumb_rev_ = tr;
        lib_.dirty = true;
      }
      if (!res.at("rev").str.empty() && res.at("rev").str != dstate_.rev) d_state(true);
      return;
    }
    if (rq.kind == "uievents") {
      for (const design::Json& e : res.at("events").arr) ui_event(e);
      return;
    }
    if (rq.kind == "quiet") return;
    if (rq.kind == "cats") {
      lib_.cats.clear();
      for (const design::Json& c : res.at("categories").arr) lib_.cats.push_back({c.s("path"), static_cast<int>(c.n("count")), static_cast<int>(c.n("depth"))});
      return;
    }
    if (rq.kind == "ranges") {
      lib_.ranges.clear();
      for (const auto& [k, r] : res.at("ranges").obj)
        if (r.arr.size() >= 2) lib_.ranges[k] = {r.arr[0].num, r.arr[1].num, r.arr.size() > 2 ? r.arr[2].str : ""};
      return;
    }
    if (rq.kind == "mypart") {  // library.my_parts: the MY PARTS count (and rows) of the workspace
      take_my_parts(res, false);
      return;
    }
    if (rq.kind == "search" && lib_.category == libui::kMyParts) {  // the MY PARTS category lists library.my_parts, not library.search
      take_my_parts(res, true);
      return;
    }
    if (rq.kind == "search") {
      lib_.hits.clear();
      lib_.total = static_cast<int>(res.n("total"));
      for (const design::Json& h : res.at("parts").arr) {
        libui::Hit x{h.s("id"), h.s("name"), h.s("vendor"), h.s("category"), h.s("thumb")};
        x.has_mass = h.at("mass").kind == design::Json::Num;
        x.mass = h.n("mass");
        lib_.hits.push_back(std::move(x));
      }
      return;
    }
    if (rq.kind == "info") {
      libui::Info& i = lib_.infos[res.s("id")];
      i = {};
      i.id = res.s("id"), i.name = res.s("name"), i.vendor = res.s("vendor"), i.category = res.s("category");
      i.part_number = res.s("part_number"), i.url = res.s("url"), i.thumb = res.s("thumb");
      for (const design::Json& p : res.at("params").arr) {
        char v[48];
        const design::Json& val = p.at("value");
        if (val.kind == design::Json::Num) std::snprintf(v, sizeof v, "%.6g", val.num);
        else std::snprintf(v, sizeof v, "%s", val.kind == design::Json::Str ? val.str.c_str() : "(table)");
        i.params.push_back({p.s("name"), p.s("unit"), p.s("fidelity"), p.s("src"), p.s("url"), p.s("quote"), v});
      }
      for (const design::Json& p : res.at("ports").arr) i.ports.push_back({p.s("name"), p.s("kind"), p.s("type"), p.s("gender")});
      for (const design::Json& b : res.at("boxes").arr) i.boxes.push_back(box_from(b));
      return;
    }
    if (rq.kind == "targets") {
      Targets& t = targets_[res.s("part")];
      t = {};
      t.requested = t.loaded = true;
      t.rev = dstate_.rev;
      for (const design::Json& b : res.at("boxes").arr) t.boxes.push_back(box_from(b));
      for (const design::Json& x : res.at("targets").arr) {
        Target g;
        g.to = x.s("to"), g.port = x.s("port");
        double v[3], r[9];
        design::vec_from(x.at("pos"), v, 3), design::vec_from(x.at("R"), r, 9);
        g.pos = {v[0], v[1], v[2]};
        std::copy(r, r + 9, g.R.m);
        design::vec_from(x.at("port_pos"), v, 3);
        g.port_pos = {v[0], v[1], v[2]};
        t.list.push_back(std::move(g));
      }
      return;
    }
    if (j.at("state").kind == design::Json::Obj) apply_state(j.at("state"));
    if (rq.kind == "suggest") {
      right_tab_ = 3;  // PLACE
      suggs_.clear();
      for (const design::Json& r : res.at("suggestions").arr)
        suggs_.push_back({r.s("part"), r.s("port"), r.s("category"), r.s("name"), r.n("score"), r.n("mass"), r.n("mass_delta"),
                          static_cast<int>(r.n("mates_kept")), static_cast<int>(r.at("dangling").arr.size()), strs(r.at("why")),
                          strs(r.at("mates_lost"))});
    } else if (rq.kind == "sim") {
      mode_ = Mode::Simulate;
      sim_title_ = res.s("kind");
      sim_rows_.clear();
      for (const auto& [k, v] : res.obj) {
        char buf[64];
        if (v.kind == design::Json::Num) std::snprintf(buf, sizeof buf, "%.5g", v.num);
        else if (v.kind == design::Json::Bool) std::snprintf(buf, sizeof buf, "%s", v.num != 0 ? "true" : "false");
        else continue;  // lists / nested tables / prose: not a quick-check number
        sim_rows_.emplace_back(k, buf);
      }
    } else if (rq.kind == "save") {
      saved_ = res.s("saved");
    } else if (rq.kind == "edit") {
      edit_done(res);
    }
  }

  void edit_done(const design::Json& res) {
    const design::Json& drc = res.at("drc");
    new_keys_.clear();
    for (const design::Json& f : drc.at("new").arr) new_keys_.insert(key(f.s("code"), strs(f.at("where"))));
    const auto counts = [](const design::Json& c) {
      return std::to_string(static_cast<int>(c.n("error"))) + " errors, " + std::to_string(static_cast<int>(c.n("warning"))) + " warnings";
    };
    // (only the delta is kept; the status bar prints the live counts in front of it and drops it once the findings have moved on)
    edit_summary_ = "(before: " + counts(drc.at("before")) + "); " + std::to_string(drc.at("new").arr.size()) + " new, " +
                    std::to_string(drc.at("resolved").arr.size()) + " resolved";
    edit_errors_ = static_cast<std::size_t>(drc.at("after").n("error")), edit_warns_ = static_cast<std::size_t>(drc.at("after").n("warning"));
    static const design::Json none;
    const design::Json& ch = res.at("changes").arr.empty() ? none : res.at("changes").arr[0];
    const std::string op = ch.s("op");
    edit_lines_.clear();
    if (op == "add" && snap_requested_) right_tab_ = 3;  // the placement alternatives of a "snap here" are on the PLACE tab
    snap_requested_ = false;
    if (op != "autowire" && op != "autowire_all") decisions_.clear();  // (a pick is an autowire: the rest of its decisions stay)
    if (op != "add") place_.alts.clear();  // cycling undoes the last edit: only valid right after the snap
    if (op == "add") {
      const design::Json& snap = ch.at("snap");
      if (cycling_ < 0 || place_.alts.empty()) {  // a fresh snap (not a cycle through the alternatives of the last one)
        place_.id = ch.s("id");
        place_.alts.clear();
        const design::Json& m = ch.at("mate");
        const auto port_of = [](const std::string& ref) { return ref.substr(ref.find('.') == std::string::npos ? ref.size() : ref.find('.') + 1); };
        place_.alts.push_back({port_of(m.s("a")), port_of(m.s("b")), static_cast<int>(snap.n("k")), snap.n("score"),
                               !snap.at("interferes").arr.empty(), strs(snap.at("why"))});
        for (const design::Json& a : snap.at("alternatives").arr)  // {to: 'inst.port', port, k, score, why, interferes}
          place_.alts.push_back({a.s("to").empty() ? a.s("a") : port_of(a.s("to")), a.s("port").empty() ? a.s("b") : a.s("port"),
                                 static_cast<int>(a.n("k")), a.n("score"), !a.at("interferes").arr.empty(), strs(a.at("why"))});
        place_.cur = 0;
      } else {
        place_.cur = cycling_;
      }
      cycling_ = -1;
      edit_lines_.push_back("added " + ch.s("id") + " (" + ch.s("part") + ")" + (ch.s("note").empty() ? "" : ": " + ch.s("note")));
    } else if (op == "swap") {
      edit_lines_.push_back("swapped " + ch.s("id") + ": " + ch.s("from") + " -> " + ch.s("to"));
      for (const std::string& d : strs(ch.at("dangling"))) edit_lines_.push_back("dangling: " + d);
      const design::Json& ad = ch.at("auto_devices");
      if (ad.kind == design::Json::Obj)
        edit_lines_.push_back("devices: " + std::to_string(ad.at("created").arr.size()) + " created, " + std::to_string(ad.at("refreshed").arr.size()) +
                              " refreshed, " + std::to_string(ad.at("unset").arr.size()) + " with unset fields");
    } else if (op == "remove") {
      edit_lines_.push_back("removed " + ch.s("id") + " (" + std::to_string(ch.at("mates_removed").arr.size()) + " mates, " +
                            std::to_string(ch.at("wires_removed").arr.size()) + " wires dropped)");
    } else if (op == "autowire") {
      const design::Json& w = ch.at("wire");
      edit_lines_.push_back("wired " + w.s("from") + " -> " + w.s("to") + " with " + w.s("part") + ", " +
                            std::to_string(w.at("length").n("value")).substr(0, 6) + " m");
      edit_lines_.push_back("why: " + ch.at("reasons").s("part"));
    } else if (op == "autowire_all") {
      for (const design::Json& w : ch.at("wired").arr)
        edit_lines_.push_back("wired " + w.at("wire").s("from") + " -> " + w.at("wire").s("to") + " (" + w.at("wire").s("part") + ")");
      decisions_.clear();
      for (const design::Json& a : ch.at("ambiguous").arr) decisions_.push_back({strs(a.at("ports")), strs(a.at("candidates")), a.s("why")});
      for (const design::Json& u : ch.at("unmatched").arr) edit_lines_.push_back("unmatched " + u.s("port") + ": " + u.s("why"));
    } else if (res.at("drc").kind == design::Json::Obj && ch.s("op").empty()) {
      edit_lines_.push_back("no change record");
    } else {
      edit_lines_.push_back(op);  // undo / redo
    }
  }

  // The backend's state after a request: shows its report, and when the built model changed loads it into the World.
  void apply_state(const design::Json& js) {
    try {
      dstate_ = design::state_from(js);
    } catch (const std::exception& e) {
      dstatus_ = std::string("state: ") + e.what();
      return;
    }
    design_ = dstate_.report;
    std::erase_if(hidden_, [&](const std::string& id) {  // a removed part's eye goes with it (a part re-added under that id starts visible)
      return std::none_of(design_.instances.begin(), design_.instances.end(), [&](const design::Instance& i) { return i.id == id; });
    });
    if (!has_design_ && mode_ == Mode::Simulate && pending_mode_.empty()) mode_ = Mode::Build;  // an editable design opens on BUILD
    has_design_ = true;
    if (std::max<std::size_t>(1, dstate_.docs.size()) != last_docs_) {  // a sub-assembly document opened / closed: a different part is on screen
      last_docs_ = std::max<std::size_t>(1, dstate_.docs.size());
      set_sel({});
      sel_sub_.clear();
      framed_ = false;
      dreq("mypart", design::request("library.my_parts"), true);  // (a saved child is a new MY PART)
    }
    rebuild_scene();
    if (!framed_ && !scene_.empty()) {  // opening a design frames it, like a CAD app (a scripted camera set before this is kept)
      framed_ = true;
      if (!cam_anim_.active && !opts_.clean) frame_selection(true, true);
    }
    for (auto& [part, t] : targets_) t.requested = false;  // the open ports changed: ghost targets are asked again on the next drag
    if (finding_ >= static_cast<int>(design_.findings.size())) {
      finding_ = -1;
      hl_.clear();
    }
    if (loaded_rev_.empty()) loaded_rev_ = dstate_.rev;  // the first state describes the model already on screen
    if (dstate_.rev != loaded_rev_ && !dstate_.xml.empty()) {
      loaded_rev_ = dstate_.rev;
      keep_camera_ = true;
      send(env_load(kSource, dstate_.xml), "load " + dstate_.xml, true);
    }
  }

  void instance_list() {
    std::map<std::string, std::vector<const design::Instance*>> by_group;
    for (const design::Instance& i : design_.instances) by_group[i.group.empty() ? "(none)" : i.group].push_back(&i);
    const design::Instance* sel = sel_instance();
    for (const auto& [g, list] : by_group) {
      if (!ImGui::TreeNodeEx(g.c_str(), ImGuiTreeNodeFlags_DefaultOpen, "%s (%zu)", g.c_str(), list.size())) continue;
      for (const design::Instance* i : list) {
        const std::string label = i->id + "   " + i->part + "   " + std::to_string(i->mass * 1000).substr(0, 7) + " g";
        const bool hl = hl_.count(i->id) > 0;
        if (hl) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1, 0.85f, 0.2f, 1});
        if (ImGui::Selectable(label.c_str(), is_selected(i->id) || sel == i)) {
          std::vector<std::string> next = ImGui::GetIO().KeyShift || ImGui::GetIO().KeyCtrl ? sel_ : std::vector<std::string>{};
          if (!std::erase(next, i->id)) next.push_back(i->id);
          set_sel(next);
        }
        if (hl) ImGui::PopStyleColor();
      }
      ImGui::TreePop();
    }
  }

  void why_tip(const std::vector<std::string>& why) const {
    if (why.empty() || !ImGui::BeginItemTooltip()) return;
    for (const std::string& w : why) ImGui::BulletText("%s", w.c_str());
    ImGui::EndTooltip();
  }

  // Library tab: parts for the selected port, snap with ranked placements, swap.
  void library_tab(const design::Instance* sel) {
    if (!sel) {
      ImGui::TextUnformatted("select an instance (list above, or a body in the scene tree)");
      return;
    }
    const std::vector<design::Port>& ports = dstate_.ports.count(sel->id) ? dstate_.ports.at(sel->id) : std::vector<design::Port>{};
    if (sel_port_inst_ != sel->id) {
      sel_port_inst_ = sel->id;
      sel_port_.clear();
    }
    ImGui::Text("%s   %s", sel->id.c_str(), sel->part.c_str());
    ImGui::SetNextItemWidth(260);
    if (ImGui::BeginCombo("port", sel_port_.empty() ? "(any open port)" : sel_port_.c_str())) {
      if (ImGui::Selectable("(any open port)", sel_port_.empty())) sel_port_.clear();
      for (const design::Port& p : ports) {
        const std::string label = p.name + "   " + p.kind + " " + p.type + (p.used ? "   (in use)" : "");
        if (ImGui::Selectable(label.c_str(), sel_port_ == p.name)) sel_port_ = p.name;
      }
      ImGui::EndCombo();
    }
    const std::string target = sel->id + (sel_port_.empty() ? "" : "." + sel_port_);
    ImGui::SetNextItemWidth(160);
    const bool enter = ImGui::InputText("category", cat_buf_, sizeof cat_buf_, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SetItemTooltip("dotted category prefix, e.g. esc, sensor.gnss, motor.bldc (empty: all)");
    ImGui::SameLine();
    if (ImGui::Button("suggest for port") || enter) d_suggest(target, cat_buf_);
    ImGui::SameLine();
    if (ImGui::Button("suggest swaps")) d_suggest(sel->id, "");
    ImGui::SetNextItemWidth(260);
    ImGui::InputText("part id", part_buf_, sizeof part_buf_);
    ImGui::SameLine();
    ImGui::BeginDisabled(!part_buf_[0]);
    if (ImGui::Button("snap here")) d_snap(part_buf_, target);
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("add this part mated to the best open port pair of %s", target.c_str());
    if (!suggs_.empty()) {
      ImGui::Separator();
      ImGui::Text("%s for %s", sugg_swap_ ? "swap candidates" : "library parts", sugg_target_.c_str());
      if (sugg_swap_) ImGui::Checkbox("refresh device bindings (auto)", &auto_dev_);
      if (ImGui::BeginTable("sugg", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable, {0, 100})) {
        ImGui::TableSetupColumn("part", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn(sugg_swap_ ? "mates kept/lost" : "port");
        ImGui::TableSetupColumn(sugg_swap_ ? "dangling" : "score");
        ImGui::TableSetupColumn(sugg_swap_ ? "d mass g" : "mass g");
        ImGui::TableSetupColumn("##go", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableHeadersRow();
        int n = 0;
        for (const Sugg& s : suggs_) {
          ImGui::PushID(n++);
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted(s.part.c_str());
          why_tip(s.why);
          if (ImGui::IsItemHovered() && !s.name.empty()) ImGui::SetTooltip("%s (%s)", s.name.c_str(), s.category.c_str());
          ImGui::TableNextColumn();
          if (sugg_swap_) ImGui::Text("%d / %zu", s.kept, s.lost.size());
          else ImGui::TextUnformatted(s.port.c_str());
          ImGui::TableNextColumn();
          if (sugg_swap_) ImGui::Text("%d", s.dangling);
          else ImGui::Text("%.0f", s.score);
          ImGui::TableNextColumn();
          ImGui::Text("%.1f", (sugg_swap_ ? s.mass_delta : s.mass) * 1000);
          ImGui::TableNextColumn();
          if (ImGui::SmallButton(sugg_swap_ ? "swap" : "snap here")) {
            if (sugg_swap_) d_swap(sugg_target_, s.part, auto_dev_);
            else d_snap(s.part, sugg_target_, "", s.port);
          }
          ImGui::PopID();
        }
        ImGui::EndTable();
      }
    }
    placement_ui();
  }

  void placement_ui() {
    if (place_.alts.empty()) return;
    ImGui::Separator();
    ImGui::Text("placement of %s on %s: %zu candidate(s)", place_.id.c_str(), place_.target.c_str(), place_.alts.size());
    ImGui::BeginDisabled(place_.alts.size() < 2);
    const int n = static_cast<int>(place_.alts.size());
    if (ImGui::SmallButton("< prev")) d_cycle((place_.cur + n - 1) % n);
    ImGui::SameLine();
    if (ImGui::SmallButton("next >")) d_cycle((place_.cur + 1) % n);
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("(each is applied as the edit; undo drops it)");
    if (ImGui::BeginTable("alts", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable, {0, 100})) {
      ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 24);
      ImGui::TableSetupColumn("ports", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("k", ImGuiTableColumnFlags_WidthFixed, 30);
      ImGui::TableSetupColumn("score", ImGuiTableColumnFlags_WidthFixed, 80);
      ImGui::TableHeadersRow();
      for (int i = 0; i < n; ++i) {
        const Alt& a = place_.alts[i];
        ImGui::PushID(i);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        if (a.interferes) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1, 0.4f, 0.4f, 1});
        if (ImGui::Selectable(std::to_string(i).c_str(), i == place_.cur, ImGuiSelectableFlags_SpanAllColumns)) {
          if (i != place_.cur) d_cycle(i);
        }
        why_tip(a.why);
        ImGui::TableNextColumn();
        ImGui::Text("%s.%s <-> %s.%s", place_.target.c_str(), a.a.c_str(), place_.id.c_str(), a.b.c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%d", a.k);
        ImGui::TableNextColumn();
        ImGui::Text("%.1f%s", a.score, a.interferes ? " X" : "");
        if (a.interferes) ImGui::PopStyleColor();
        ImGui::PopID();
      }
      ImGui::EndTable();
    }
  }

  void wiring_tab(const design::Instance* sel) {
    const std::string port = sel && !sel_port_.empty() && sel_port_inst_ == sel->id ? sel->id + "." + sel_port_ : "";
    ImGui::BeginDisabled(port.empty());
    if (ImGui::Button(("autowire " + (port.empty() ? std::string("selected port") : port)).c_str())) d_autowire(port);
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("pick the port in the Library tab; wire part and length are chosen by the Design API");
    ImGui::SameLine();
    if (ImGui::Button("autowire all")) d_autowire_all();
    ImGui::SetItemTooltip("wires every open port that has one sensible target; the ambiguous ones are listed for you to pick");
    if (!decisions_.empty()) {
      ImGui::Separator();
      ImGui::TextColored({1, 0.8f, 0.3f, 1}, "%zu decision(s) for you (nothing was guessed)", decisions_.size());
    }
    for (std::size_t di = 0; di < decisions_.size(); ++di) {
      Decision& d = decisions_[di];
      if (d.ports.empty()) continue;
      ImGui::PushID(static_cast<int>(di));
      ImGui::TextWrapped("%s", d.why.c_str());
      for (std::size_t pi = 0; pi < d.ports.size(); ++pi) {
        ImGui::PushID(static_cast<int>(pi));
        ImGui::SetNextItemWidth(220);
        if (ImGui::BeginCombo(("##" + d.ports[pi]).c_str(), ("wire " + d.ports[pi] + " to ...").c_str())) {
          for (const std::string& c : d.cands)
            if (ImGui::Selectable(c.c_str())) {
              d_autowire(d.ports[pi], c);
              const std::string p = d.ports[pi];
              d.cands.erase(std::remove(d.cands.begin(), d.cands.end(), c), d.cands.end());  // taken
              d.ports.erase(d.ports.begin() + static_cast<long>(pi));
              ImGui::EndCombo();
              ImGui::PopID();
              ImGui::PopID();
              return;
            }
          ImGui::EndCombo();
        }
        ImGui::PopID();
      }
      ImGui::PopID();
    }
  }

  void simulate_tab() {
    if (ImGui::Button("hover")) d_simulate("hover");
    ImGui::SetItemTooltip("multirotor: built-in altitude + attitude controller, 2 m for 10 s (the Design API's simulate)");
    ImGui::SameLine();
    if (ImGui::Button("hold")) d_simulate("hold");
    ImGui::SetItemTooltip("arm: servos torque-on at a pose, reports sag and current");
    if (dfut_.valid() && dcur_.kind == "sim") ImGui::TextUnformatted("running...");
    else if (!sim_title_.empty()) ImGui::Text("last: %s", sim_title_.c_str());
    if (ImGui::BeginTable("sim", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, {0, 220})) {
      for (const auto& [k, v] : sim_rows_) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(k.c_str());
        ImGui::TableNextColumn();
        const bool bad = k == "pass" && v == "false";
        if (bad) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1, 0.35f, 0.35f, 1});
        ImGui::TextUnformatted(v.c_str());
        if (bad) ImGui::PopStyleColor();
      }
      ImGui::EndTable();
    }
  }

  // ================================ layout: top bar, status bar, per-mode panes =====================================================
  enum class Mode { Build, Library, Wiring, Simulate, Program };
  // kApp = the application bar; kTop = where the page content starts (below the ribbon); kStatus = the status bar.
  static constexpr float kApp = theme::kAppBar, kTop = theme::kAppBar + theme::kRibbon, kStatus = theme::kStatusBar;
  static constexpr ImGuiWindowFlags kPane = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav;

  static const char* mode_name(Mode m) {
    static const char* const n[] = {"build", "library", "wiring", "simulate", "program"};
    return n[static_cast<int>(m)];
  }
  bool set_mode(const std::string& name) {
    for (Mode m : {Mode::Build, Mode::Library, Mode::Wiring, Mode::Simulate, Mode::Program})
      if (name == mode_name(m)) {
        if (!has_design_ && m != Mode::Simulate && m != Mode::Program) return false;
        if (m != Mode::Simulate && m != Mode::Program) run_.restore();  // leaving for the design pages: its own model back in the World
        mode_ = m;
        return true;
      }
    return false;
  }

  // The viewport = the part of the window the panes leave free; the 3D scene is rendered full-window and the camera is shifted so
  // its target sits in the middle of the viewport (WindowRenderer draws the whole swapchain image).
  void layout(float W, float H) {
    const bool narrow = W < 900;
    float l = 0, r = 0;
    switch (mode_) {
      case Mode::Build: l = narrow ? 0 : 268, r = narrow ? 0 : 316; break;
      case Mode::Library: l = std::round(W * 0.78f); break;
      case Mode::Simulate: l = narrow ? 0 : 290, r = narrow ? 0 : 290; break;
      default: break;
    }
    pane_l_ = l;
    pane_r_ = r;
    if (opts_.clean) {  // HUD / recording: the whole window is the viewport
      pane_l_ = pane_r_ = 0;
      full_page_ = false;
      vp_goal_ = vp_ = {0, 0, W, H};  // (vp_ eases toward vp_goal_ each frame: set both)
      return;
    }
    full_page_ = mode_ == Mode::Wiring || mode_ == Mode::Program;
    vp_goal_ = full_page_ ? ImVec4{0, kTop, W, H - kStatus} : mode_ == Mode::Library ? ImVec4{l, kTop, W, H - kStatus} : ImVec4{l, kTop, W - r, H - kStatus};
  }
  // The 3D viewport follows its goal with an ease (~0.2 s), so the scene glides when a mode change opens or closes side panels.
  void smooth_viewport(float dt) {
    const float k = vp_init_ ? 1.f - std::exp(-14.f * std::min(dt, 0.1f)) : 1.f;
    const float* g = &vp_goal_.x;
    float* v = &vp_.x;
    for (int i = 0; i < 4; ++i) v[i] = std::abs(g[i] - v[i]) < 0.5f ? g[i] : v[i] + (g[i] - v[i]) * k;
    vp_init_ = true;
  }
  Camera view_camera(double W, double H) const {
    Camera c = orbit_.camera();
    if (full_page_) return c;
    const double tanh = std::tan(c.fovy * 3.14159265358979 / 360), aspect = W / H;
    const double offx = 0.5 * (vp_.x + vp_.z) - 0.5 * W, offy = 0.5 * (vp_.y + vp_.w) - 0.5 * H;  // px the target must move
    const b3::Cam v = b3::Cam::make(c, W, H);
    const b3::V3 shift = v.r * (-offx / (0.5 * W) * orbit_.dist * tanh * aspect) + v.u * (offy / (0.5 * H) * orbit_.dist * tanh);
    for (int k = 0; k < 3; ++k) {
      c.pos[k] += shift[k];
      c.lookat[k] += shift[k];
    }
    return c;
  }

  // The application bar: mark + design name on the left, the workspace (page) tabs, quick-access undo / redo / save on the right.
  void top_bar(float W) {
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({W, kApp});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kBar);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16, 0});
    ImGui::Begin("##top", nullptr, kPane | ImGuiWindowFlags_NoScrollbar);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled({12, 8}, {30, 26}, theme::u32(theme::kAccent), 3);  // the mark: an isometric cube in the accent square
    theme::icon(dl, theme::Icon::Cube, {21, 17}, 14, IM_COL32(255, 255, 255, 255));
    ImGui::PushFont(nullptr, theme::kFontTitle);
    dl->AddText({38, 8}, theme::u32(theme::kText), "DVSIM");
    ImGui::PopFont();
    // The document title as a breadcrumb: the base design, then every open sub-assembly document ("dog > fl (leg.toml)"). A dirty
    // document carries a "*". Clicking an earlier crumb returns to it (close_docs: a dirty child asks to save first).
    struct Crumb {
      std::string label;
      bool dirty;
    };
    std::vector<Crumb> crumbs;
    if (dstate_.docs.size() > 1)
      for (const design::DocRow& dr : dstate_.docs) crumbs.push_back({dr.crumb.empty() ? dr.title : dr.crumb.back(), dr.dirty});
    else crumbs.push_back({has_design_ && !design_.assembly.empty() ? design_.assembly : sim_title_.empty() ? "untitled" : sim_title_,
                           dstate_.docs.empty() ? dstate_.undo > 0 : dstate_.docs[0].dirty});  // (undo stack not empty: unsaved edits)
    ImGui::PushFont(nullptr, theme::kFontSmall);
    float title_w = 0;
    rep_.crumbs.clear();
    if (W >= 1000) {
      float cx = 90;
      const float cmax = 90 + 360;  // the crumbs never run into the page tabs
      for (std::size_t k = 0; k < crumbs.size() && cx < cmax; ++k) {
        const bool last = k + 1 == crumbs.size();
        const std::string lab = crumbs[k].label + (crumbs[k].dirty ? " *" : "");
        const float w = std::min(ImGui::CalcTextSize(lab.c_str()).x, last ? 220.f : 110.f);
        bool hov = false;
        if (!last) {
          ImGui::SetCursorScreenPos({cx, 6});
          ImGui::PushID(static_cast<int>(k));
          if (ImGui::InvisibleButton("##crumb", {w + 4, kApp - 12})) close_docs(static_cast<int>(k));
          ImGui::PopID();
          hov = ImGui::IsItemHovered();
          ImGui::SetItemTooltip("back to %s", crumbs[k].label.c_str());
        }
        dl->PushClipRect({cx, 0}, {cx + w + 2, kApp}, true);
        dl->AddText({cx, 11}, theme::u32(hov || (last && crumbs.size() > 1) ? theme::kText : theme::kDim), lab.c_str());
        dl->PopClipRect();
        rep_.crumbs.push_back({lab, {cx, 6, cx + w + 4, kApp - 6}});
        cx += w + 4;
        if (!last) {
          dl->AddText({cx + 2, 11}, theme::u32(theme::kDim, 0.6f), ">");
          cx += 18;
        }
      }
      title_w = cx - 90 + 24;
    }
    ImGui::PopFont();
    ImGui::SetCursorScreenPos({90 + title_w + 8, 4});
    std::vector<const char*> labels;
    std::vector<Mode> modes;
    for (Mode m : {Mode::Build, Mode::Library, Mode::Wiring, Mode::Simulate, Mode::Program}) {
      if (!has_design_ && m != Mode::Simulate && m != Mode::Program) continue;
      static const char* const caps[] = {"BUILD", "LIBRARY", "WIRING", "SIMULATE", "PROGRAM"};
      labels.push_back(caps[static_cast<int>(m)]);
      modes.push_back(m);
    }
    int active = 0;
    for (std::size_t i = 0; i < modes.size(); ++i)
      if (modes[i] == mode_) active = static_cast<int>(i);
    const float tabs_x0 = ImGui::GetCursorScreenPos().x;
    const int hit = theme::tabs(labels.data(), static_cast<int>(labels.size()), active);
    rep_.tabs = {tabs_x0, 0, ImGui::GetCursorScreenPos().x - 26.f, kApp};  // (theme::tabs leaves the cursor one default gap past the last tab)
    if (hit >= 0) set_mode(mode_name(modes[hit]));
    // right edge: [zoom] [keys] [undo redo save]; each piece stays left of the one to its right (rects exported in rep_)
    const float quick_w = opts_.design_api ? 3 * 28.f : 0.f, keys_x = W - 12 - quick_w - 28;
    rep_.zoom = {0, 0, 0, 0};
    if (mode_ == Mode::Wiring && flat_.zoom_percent() > 0) {  // the 2D canvas's zoom; a click fits the sheet in the window
      char z[16];
      std::snprintf(z, sizeof z, "%d%%", flat_.zoom_percent());
      ImGui::PushFont(nullptr, theme::kFontSmall);
      const ImVec2 ts = ImGui::CalcTextSize(z);
      const float zx = keys_x - 22 - ts.x;
      ImGui::SetCursorScreenPos({zx - 8, 6});
      if (ImGui::InvisibleButton("##wirezoom", {ts.x + 16, kApp - 12})) flat_.fit_view();
      rep_.zoom = {zx - 8, 6, zx + ts.x + 8, kApp - 6};
      dl->AddText({zx, (kApp - ts.y) / 2}, theme::u32(ImGui::IsItemHovered() ? theme::kText : theme::kDim), z);
      ImGui::PopFont();
      ImGui::SetItemTooltip("zoom: click to fit the sheet");
    }
    ImGui::SetCursorScreenPos({keys_x, 3});  // the controls popover (controls_popover, drawn after the page)
    if (theme::icon_button("##qkeys", theme::Icon::Keys, 28, keys_open_, true, "controls: mouse and keys of this page (F1)")) keys_open_ = !keys_open_;
    rep_.icon = {keys_x, 3, keys_x + 28, 31};
    if (opts_.design_api) {  // quick access (right): undo / redo / save
      using theme::Icon;
      float x = W - 12 - 3 * 28;
      ImGui::SetCursorScreenPos({x, 3});
      if (theme::icon_button("##qundo", Icon::Undo, 28, false, dstate_.undo > 0, "undo (Ctrl+Z)")) d_undo(false);
      ImGui::SetCursorScreenPos({x + 28, 3});
      if (theme::icon_button("##qredo", Icon::Redo, 28, false, dstate_.redo > 0, "redo (Ctrl+Shift+Z)")) d_undo(true);
      ImGui::SetCursorScreenPos({x + 56, 3});
      if (theme::icon_button("##qsave", Icon::Save, 28, false, true, nullptr)) d_save(save_buf_);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("save: write the design TOML%s%s", dstate_.path.empty() ? "" : ": ", dstate_.path.c_str());
      if (save_buf_[0] == 0 && !dstate_.path.empty()) std::snprintf(save_buf_, sizeof save_buf_, "%s", dstate_.path.c_str());
    }
    ImGui::Dummy({0, 0});  // (theme::tabs leaves the cursor past its last tab: an item must follow)
    dl->AddLine({0, kApp - 0.5f}, {W, kApp - 0.5f}, theme::u32(theme::kLine));
    ImGui::End();
  }

  // The ribbon under the application bar: grouped icon + label tools of the active page. Every button is an existing function of the
  // window (gizmo mode, the Backend ops, the page's own actions); nothing here is decoration.
  void ribbon_bar(float W) {
    ImGui::SetNextWindowPos({0, kApp});
    ImGui::SetNextWindowSize({W, theme::kRibbon});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kPanel);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::Begin("##ribbon", nullptr, kPane | ImGuiWindowFlags_NoScrollbar);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    using theme::Icon;
    theme::Ribbon r;
    r.begin({0, kApp});
    const bool ed = opts_.design_api && has_design_;
    const design::Instance* sel = sel_instance();
    const bool can = ed && sel && sel->id != dstate_.root;
    const auto history = [&] {
      r.group("HISTORY");
      if (r.button(Icon::Undo, "Undo", false, ed && dstate_.undo > 0, "undo the last edit (Ctrl+Z)")) d_undo(false);
      if (r.button(Icon::Redo, "Redo", false, ed && dstate_.redo > 0, "redo (Ctrl+Shift+Z)")) d_undo(true);
    };
    switch (mode_) {
      case Mode::Build: {
        r.group("SELECT");
        if (r.button(Icon::Select, "Select", gizmo_.mode == b3::Gizmo::Off, ed, "no gizmo: click to select, Shift/Ctrl adds")) gizmo_.mode = b3::Gizmo::Off, measure_ = 0;
        if (r.button(Icon::Move, "Move", gizmo_.mode == b3::Gizmo::Move, ed, "G: move along an axis or on the ground plane")) gizmo_.mode = b3::Gizmo::Move, measure_ = 0;
        if (r.button(Icon::Rotate, "Rotate", gizmo_.mode == b3::Gizmo::Rotate, ed, "R: rotate about an axis (snaps to the angle step)")) gizmo_.mode = b3::Gizmo::Rotate, measure_ = 0;
        r.group("MODIFY");
        if (r.button(Icon::Align, "Align", false, ed && sel_.size() > 1, "align the selected parts (pick two or more): opens the ALIGN list")) ImGui::OpenPopup("##align");
        const std::string msub = sel ? sub_of(sel->id) : sel_sub_;  // a sub-assembly (or one of its items) is selected
        if (r.button(Icon::Mirror, "Mirror", false, can || !msub.empty(), "mirror a copy across the world plane normal to Y; for a sub-assembly: a mirrored copy (pick the port it attaches to in the Library tab first)")) {
          if (!msub.empty()) {
            const std::string to = !sel_port_.empty() && sub_of(sel_port_inst_).empty() ? sel_port_inst_ + "." + sel_port_ : "";
            sub_req("sub.mirror", "{\"id\":" + q(msub) + ",\"plane\":\"xz\"" + (to.empty() ? "" : ",\"to\":" + q(to)) + "}");
          } else {
            dreq("edit", design::request("mirror", "{\"silent\":true,\"instances\":" + json_list(sel_) + ",\"axis\":\"y\"}"));
          }
        }
        if (r.button(Icon::Copy, "Duplicate", false, can, "copy the selected part next to itself")) dreq("edit", design::request("duplicate", "{\"silent\":true,\"instance\":" + q(sel->id) + "}"));
        if (r.button(Icon::Trash, "Remove", false, can, "Del: remove the selected part")) d_remove(sel->id);
        r.group("ASSEMBLE");
        if (r.button(Icon::Mate, "Mate", snap_.ports, ed, "snap to ports while dragging (mate seats); the PLACE tab lists the alternatives")) snap_.ports = !snap_.ports;
        if (r.button(Icon::Colour, "Colour", false, can, "pick the part's colour")) {
          right_tab_ = 0;
          color_flash_ = ImGui::GetTime() + 0.05;
        }
        r.group("INSPECT");
        if (r.button(Icon::Measure, "Measure", measure_ > 0, ed, "click two parts: distance between their centres and the clear gap (Esc ends)")) measure_ = measure_ ? 0 : 1, measure_a_.clear(), measure_b_.clear();
        if (r.button(Icon::Check, "Check", right_tab_ == 2, ed, "design rule check: findings on the DRC tab")) right_tab_ = 2;
        history();
        r.group("DOCUMENT");
        if (r.button(Icon::Save, "Save", false, ed, "write the design TOML")) d_save(save_buf_);
        if (r.button(Icon::Doc, "New part", false, ed, "start a new part (sub-assembly) in its own document; Export as... in the inspector, MY PARTS in the Library")) {
          std::string n;
          do n = "part" + std::to_string(++part_seq_);
          while (find_sub(n));
          sub_req("sub.new", "{\"name\":" + q(n) + "}");
        }
        break;
      }
      case Mode::Library: {
        r.group("FIND");
        if (r.button(Icon::Clear, "Clear", false, true, "clear the search text and category")) {
          lib_.query[0] = 0;
          lib_.category.clear();
          lib_.dirty = true;
        }
        r.group("USE");
        const bool pick = !lib_.selected.empty() && ed;
        if (r.button(Icon::Add, "Add", false, pick, "add the open part to the design (onto the selected part's best open port)")) add_part(lib_.selected);
        if (r.button(Icon::Swap, "Swap", false, pick && sel, "replace the selected part with the open one")) d_swap(sel->id, lib_.selected, true);
        history();
        break;
      }
      case Mode::Wiring: {
        r.group("WIRE");
        const bool fl = flat_.active();
        if (r.button(Icon::Wire, "Wire", fl, ed, "2D canvas: drag from a pin to another pin to lay a wire")) flat_.begin_selected(dstate_, sel_.empty() ? "" : sel_.back());
        if (r.button(Icon::Pin, "Pin wire", fl, ed && fl, "pin-level wires: a connector's pins are drawn one by one; drag pin to pin")) flat_.zoom_to_pins();
        r.group("AUTOMATE");
        if (r.button(Icon::Auto, "Autowire", false, ed, "wire every open port that has one sensible target")) d_autowire_all();
        r.group("VERIFY");
        if (r.button(Icon::Check, "DRC", false, ed, "design rule check (DRC tab in BUILD)")) set_mode("build"), right_tab_ = 2;
        if (r.button(Icon::Fit, "Fit", false, fl, "fit the sheet in the window")) flat_.fit_view();
        history();
        break;
      }
      case Mode::Simulate: {
        r.group("RUN");
        if (run_.live()) {
          if (r.button(Icon::Clear, "Stop", true, true, "stop the live program (kills it)")) run_.toggle();
        } else if (cur_.running) {
          if (r.button(Icon::Pause, "Pause", true, true, "pause the simulation")) send(env_pause(kSource), "pause");
        } else if (r.button(Icon::Play, "Run", false, true, "run at the rate set in the left panel")) {
          send(env_run(kSource, rate_), "run");
        }
        if (r.button(Icon::Step, "Step", false, !cur_.running && !run_.live(), "advance by the ticks set in the left panel")) send(env_step(kSource, std::max(1, steps_)), "step");
        r.group("VIEW");
        if (r.button(Icon::Fit, "Fit", false, !scene_.empty(), "frame the whole model")) frame_selection(true);
        if (r.button(Icon::Home, "Iso", false, true, "isometric view")) camera_preset("iso");
        if (ed) {
          r.group("DESIGN CHECK");
          if (r.button(Icon::Play, "Hover", false, true, "design check run: hover 5 s")) d_simulate("hover");
        }
        break;
      }
      case Mode::Program: {
        r.group("FILE");
        if (r.button(Icon::Doc, "Design", false, true, "open the design TOML (read-only)")) prog_.command("design");
        if (r.button(Icon::Save, "Save", false, !prog_.active_path().empty(), "save the open file (Ctrl+S)")) prog_.command("save");
        if (r.button(Icon::Copy, "Save all", false, prog_.any_dirty(), "save every modified file (Ctrl+Shift+S)")) prog_.command("save_all");
        r.group("NAVIGATE");
        if (r.button(Icon::Search, "Open", false, true, "quick open a file (Ctrl+P)")) prog_.command("quick_open");
        if (r.button(Icon::Search, "Find", false, !prog_.active_path().empty(), "find / replace in the file (Ctrl+F, Ctrl+H)")) prog_.command("find");
        if (r.button(Icon::Net, "In files", false, true, "find in files (Ctrl+Shift+F)")) prog_.command("find_in_files");
        if (r.button(Icon::Select, "Go to line", false, !prog_.active_path().empty(), "go to line (Ctrl+G)")) prog_.command("goto_line");
        r.group("TERMINAL");
        if (r.button(Icon::Add, "New", false, true, "open another shell tab")) prog_.command("new_terminal");
        if (r.button(Icon::Clear, "Clear", false, true, "empty the terminal scrollback (Ctrl+L)")) prog_.command("clear_terminal");
        if (ed) history();
        break;
      }
    }
    r.end();
    if (ImGui::BeginPopup("##align")) {  // ALIGN: min / centre / max of the selection on an axis, or equal gaps
      const char* const axes[] = {"X", "Y", "Z"};
      const char* const to[] = {"min", "center", "max"};
      for (int ax = 0; ax < 3; ++ax) {
        ImGui::TextUnformatted(axes[ax]);
        for (int t = 0; t < 3; ++t) {
          ImGui::SameLine();
          ImGui::PushID(ax * 3 + t);
          if (ImGui::SmallButton(to[t])) place_group("align", ax, to[t]), ImGui::CloseCurrentPopup();
          ImGui::PopID();
        }
        ImGui::SameLine();
        ImGui::PushID(100 + ax);
        if (ImGui::SmallButton("distribute")) place_group("distribute", ax, ""), ImGui::CloseCurrentPopup();
        ImGui::PopID();
      }
      ImGui::EndPopup();
    }
    ImGui::End();
  }

  void status_bar(float W, float H) {
    ImGui::SetNextWindowPos({0, H - kStatus});
    ImGui::SetNextWindowSize({W, kStatus});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kBar);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {12, 0});
    ImGui::Begin("##status", nullptr, kPane | ImGuiWindowFlags_NoScrollbar);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddLine({0, H - kStatus + 0.5f}, {W, H - kStatus + 0.5f}, theme::u32(theme::kLine));
    ImGui::PushFont(nullptr, theme::kFontSmall);
    const float y = H - kStatus + 5;
    std::size_t errors = 0, warns = 0;
    for (const design::Finding& f : design_.findings) {
      if (f.severity == "error") ++errors;
      else if (f.severity == "warning") ++warns;
    }
    // right-hand readouts, right to left; each one is dropped (lowest priority first) when the window is too narrow for it
    struct Seg {
      std::string text;
      ImVec4 col;
      int prio;  // dropped first when high
    };
    std::vector<Seg> segs;
    char b[96];
    std::snprintf(b, sizeof b, "T %.3f s", static_cast<double>(cur_.tick) * cur_.dt);
    segs.push_back({b, theme::kDim, 0});
    if (has_design_) {
      std::snprintf(b, sizeof b, "DRC  %zu ERR  %zu WARN", errors, warns);
      segs.push_back({b, errors ? theme::kErr : warns ? theme::kWarn : theme::kOk, 0});
      std::snprintf(b, sizeof b, "MASS  %.3f kg", design_.mass_total);
      segs.push_back({b, theme::kText, 1});
      rep_.cost.clear();
      if (design_.has_bom && design_.bom_cost >= 0) {  // report["bom"].total.cost: a lower bound while some parts have no price
        std::snprintf(b, sizeof b, "COST  $%.2f%s", design_.bom_cost, design_.bom_missing > 0 ? "+" : "");
        rep_.cost = b;
        segs.push_back({b, theme::kText, 1});
      }
      std::snprintf(b, sizeof b, "%zu PARTS", design_.instances.size());
      segs.push_back({b, theme::kText, 2});
      if (snap_.grid > 0) std::snprintf(b, sizeof b, "SNAP  %g mm  %g\xC2\xB0%s%s", snap_.grid * 1000, snap_.angle, snap_.edges ? "  EDGE" : "", snap_.ports ? "  PORT" : "");
      else std::snprintf(b, sizeof b, "SNAP  OFF");
      segs.push_back({b, snap_.grid > 0 || snap_.edges || snap_.ports ? theme::kText : theme::kDim, 3});
      segs.push_back({"UNITS  mm  kg", theme::kDim, 4});
    }
    std::string left = "NO SELECTION";
    if (!sel_.empty()) {
      left = "SELECTED  " + sel_.back();
      if (const design::Instance* i = sel_instance()) {
        left += "  " + i->part;
        if (!sel_port_.empty() && sel_port_inst_ == i->id) left += "  ." + sel_port_;  // the pinout picked in the Library tab
      }
      if (sel_.size() > 1) left += "  +" + std::to_string(sel_.size() - 1);
    } else if (!selected_.empty()) {
      left = "PRIM  " + selected_;
    }
    const float min_left = 300, sep = 26;
    const auto total = [&] {
      float w = 12;
      for (const Seg& s : segs) w += ImGui::CalcTextSize(s.text.c_str()).x + sep;
      return w;
    };
    while (segs.size() > 1 && W - total() < min_left) {  // drop the highest-priority-number segment
      auto it = std::max_element(segs.begin(), segs.end(), [](const Seg& a, const Seg& c) { return a.prio < c.prio; });
      segs.erase(it);
    }
    float x = W - 12;
    for (const Seg& s : segs) {  // drawn right to left in the order pushed: time at the far right
      x -= ImGui::CalcTextSize(s.text.c_str()).x;
      dl->AddText({x, y}, theme::u32(s.col), s.text.c_str());
      x -= sep / 2;
      dl->AddLine({x, H - kStatus + 5}, {x, H - 5}, theme::u32(theme::kLine));
      x -= sep / 2;
    }
    // left: the selection, then the message / hint of the page in what is left
    const float left_w = std::min(ImGui::CalcTextSize(left.c_str()).x, (x - 12) * 0.5f);
    dl->PushClipRect({12, 0}, {12 + left_w, H}, true);
    dl->AddText({12, y}, theme::u32(theme::kText), left.c_str());
    dl->PopClipRect();
    std::string msg = !dstatus_.empty() ? dstatus_ : !dstate_.build_error.empty() ? "does not build: " + dstate_.build_error : !status_.empty() ? status_ : dfut_.valid() && !dcur_.quiet ? "working: " + dcur_.kind + " ..." : "";
    if (msg.empty() && !edit_summary_.empty()) {  // the last edit's DRC delta, headed by the same counts the chip shows
      if (errors == edit_errors_ && warns == edit_warns_)
        msg = "DRC " + std::to_string(errors) + " errors, " + std::to_string(warns) + " warnings " + edit_summary_;
      else edit_summary_.clear();  // the findings moved on since that edit (another client, a rebuild, a pin-level wire): it is stale
    }
    status_msg_ = msg;
    chip_errors_ = errors, chip_warns_ = warns;
    const bool bad = !dstatus_.empty() || !status_.empty() || !dstate_.build_error.empty();
    const float mx = 12 + left_w + 24, avail = x - 12 - mx;  // (the controls of each page live in the top bar's popover, not here)
    if (!msg.empty() && avail > 40) {
      if (ImGui::CalcTextSize(msg.c_str()).x > avail) {  // a message that does not fit gets an ellipsis
        while (msg.size() > 1 && ImGui::CalcTextSize((msg + "...").c_str()).x > avail) msg.pop_back();
        msg += "...";
      }
      if (!msg.empty()) dl->AddText({mx, y}, theme::u32(bad ? theme::kWarn : theme::kDim, bad ? 1.f : 0.8f), msg.c_str());
    }
    ImGui::PopFont();
    ImGui::End();
  }

  ImVec2 pane_pos(bool right, float W) const { return {right ? W - pane_r_ : 0, kTop}; }
  void begin_pane(const char* id, ImVec2 pos, ImVec2 size, bool solid = false) {
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, solid ? theme::kBg : theme::kPanel);
    ImGui::Begin(id, nullptr, kPane);
    ImGui::PopStyleColor();
  }

  // ---------------------------------------- BUILD ------------------------------------------------------------------------------
  void build_panes(float W, float H) {
    const float ph = H - kTop - kStatus;
    if (pane_l_ > 0) browser_pane(ph);
    if (pane_r_ > 0) inspector_pane(W, ph);
    viewport_overlay();
  }

  // The bindings of the current page: {key, effect}. (PROGRAM: the chords of program_ui.cpp's shortcut block and terminal; WIRING: the
  // canvas bindings of wire2d.cpp canvas_ui; the others: viewport_layer / shortcuts here.)
  struct KeyRow {
    const char *key, *what;
  };
  std::vector<KeyRow> key_rows() const {
    switch (mode_) {
      case Mode::Build:
        return {{"drag", "orbit"}, {"right drag", "pan"}, {"wheel", "zoom"}, {"click", "select"}, {"G  R", "move, rotate"}, {"F", "frame selection"},
                {"1-4", "front side top iso"}, {"Del", "remove"}, {"Ctrl+Z", "undo (+Shift redo)"}, {"Ctrl+D", "duplicate"}};
      case Mode::Library:
        return {{"drag card", "drop in the 3D view"}, {"click card", "open the part"}, {"drag", "orbit the view"}, {"wheel", "zoom"}};
      case Mode::Wiring:
        return {{"drag pin", "wire to a pin"}, {"drag header", "move the card"}, {"wheel", "zoom"}, {"middle drag", "pan"},
                {"click wire", "select it"}, {"Del", "remove the wire"}};
      case Mode::Simulate:
        return {{"drag", "orbit"}, {"right drag", "pan"}, {"wheel", "zoom"}};
      case Mode::Program:
        return {{"Ctrl+P", "open a file"},     {"Ctrl+S", "save"},         {"Ctrl+Shift+S", "save all"},      {"Ctrl+F", "find"},
                {"Ctrl+H", "find, replace"},   {"Ctrl+G", "go to line"},   {"Ctrl+Shift+F", "find in files"}, {"F3", "next match (+Shift prev)"},
                {"F12", "definition (python)"}, {"F2", "rename (python)"}, {"Ctrl+W", "close tab"},            {"Ctrl+Shift+T", "reopen tab"},
                {"Ctrl+`", "terminal"},         {"Ctrl+L", "clear terminal"}};
    }
    return {};
  }

  // F1 toggles the popover, Esc closes it. Only when no text field has the keyboard: typing F1 / Esc in the PROGRAM editor, the
  // terminal or the library search belongs to that field.
  void popover_keys() {
    esc_used_ = false;
    if (ImGui::GetIO().WantTextInput) return;
    if (ImGui::IsKeyPressed(ImGuiKey_F1, false)) keys_open_ = !keys_open_;
    else if (keys_open_ && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) keys_open_ = false, esc_used_ = true;
  }

  // The controls popover under the keys icon of the top bar (the legend that used to sit on the viewport). It is rebuilt every frame
  // from key_rows(), so it always shows the page that is up. A click anywhere else closes it.
  void controls_popover() {
    rep_.pop = {0, 0, 0, 0};
    if (!keys_open_) return;
    const std::vector<KeyRow> rows = key_rows();
    constexpr float kW = 270, kHead = 26, kRow = 17;
    static const char* const names[] = {"BUILD", "LIBRARY", "WIRING", "SIMULATE", "PROGRAM"};
    const float h = kHead + 6 + static_cast<float>(rows.size()) * kRow + 4;
    const ImVec2 pos{std::max(4.f, rep_.icon.z - kW), kApp + 2};
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize({kW, h});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4{0.03f, 0.04f, 0.055f, 1.f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.f);
    // (not kPane: the docked panes must stay under it, so it keeps the default bring-to-front when it appears)
    ImGui::Begin("##keyspop", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());  // (windows that appeared later this frame, or on a page switch, stay under it)
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImU32 dim = theme::u32(theme::kDim);
    dl->AddRect(pos, {pos.x + kW, pos.y + h}, theme::u32(theme::kLine), 6);
    ImGui::PushFont(nullptr, theme::kFontSmall);
    dl->AddText({pos.x + 10, pos.y + 6}, dim, (std::string("CONTROLS  ") + names[static_cast<int>(mode_)]).c_str());
    float y = pos.y + kHead + 6;
    for (const KeyRow& r : rows) {
      dl->AddText({pos.x + 10, y}, theme::u32(theme::kText), r.key);
      dl->AddText({pos.x + 100, y}, dim, r.what);
      y += kRow;
    }
    ImGui::PopFont();
    rep_.pop = {pos.x, pos.y, pos.x + kW, pos.y + h};
    const ImVec2 mp = ImGui::GetIO().MousePos;
    const bool on_icon = mp.x >= rep_.icon.x && mp.x <= rep_.icon.z && mp.y >= rep_.icon.y && mp.y <= rep_.icon.w;  // (the icon toggles it itself)
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) && !on_icon) keys_open_ = false;
    ImGui::End();
  }

  // What the window shows, as JSON for the backend (ui.report -> ui.state.layout): the page, the popover and its rows, the status
  // text and the DRC chip, the rectangles of the top bar pieces and the live ImGui windows. Sent only when it changes. Tests and
  // agents read it; nothing in the window depends on the answer.
  void report_ui() {
    if (!opts_.design_api) return;
    const auto rect = [](const ImVec4& r) {
      char b[96];
      std::snprintf(b, sizeof b, "[%.1f,%.1f,%.1f,%.1f]", static_cast<double>(r.x), static_cast<double>(r.y), static_cast<double>(r.z), static_cast<double>(r.w));
      return std::string(b);
    };
    std::string rows, wins;
    if (keys_open_)
      for (const KeyRow& r : key_rows()) rows += (rows.empty() ? "" : ",") + q(std::string(r.key) + " | " + r.what);
    for (const ImGuiWindow* w : ImGui::GetCurrentContext()->Windows)
      if (w->Active && w->Name[0] == '#') wins += (wins.empty() ? "" : ",") + q(w->Name);
    char b[64];
    std::snprintf(b, sizeof b, "{\"errors\":%zu,\"warnings\":%zu}", chip_errors_, chip_warns_);
    // sub-assembly workspace: breadcrumb, tree nodes, exports, MY PARTS, the inspector's "from" line, cost and the per-sub panel
    std::string crumbs, tree, exps, mine, panel, hits;
    for (const Rep::CrumbRect& c : rep_.crumbs) crumbs += (crumbs.empty() ? "" : ",") + q(c.label);
    for (const SubNode& sn : sub_nodes())
      tree += (tree.empty() ? "" : ",") + ("{\"id\":" + q(sn.id) + ",\"file\":" + q(sn.file) + ",\"mirror\":" + q(sn.mirror) + ",\"error\":" + q(sn.error) +
                                           ",\"items\":" + std::to_string(sn.items.size()) + "}");
    for (const design::ExportRow& e : dstate_.exports) exps += (exps.empty() ? "" : ",") + q(e.kind + ":" + e.name + "=" + e.ref);
    for (const auto& [path, m] : lib_.mine) mine += (mine.empty() ? "" : ",") + q(m.name);
    for (const std::string& p : rep_.panel) panel += (panel.empty() ? "" : ",") + q(p);
    for (const libui::Hit& h : lib_.hits) hits += (hits.empty() ? "" : ",") + q(h.name);
    const std::string subj = ",\"crumbs\":[" + crumbs + "],\"tree\":[" + tree + "],\"exports\":[" + exps + "],\"my_parts\":[" + mine + "],\"hits\":[" + hits +
                             "],\"category\":" + q(lib_.category) + ",\"from\":" + q(rep_.from) + ",\"cost\":" + q(rep_.cost) + ",\"panel\":[" + panel +
                             "],\"confirm\":" + q(close_ask_.open ? "close_doc" : "") + ",\"export_open\":" + (ImGui::IsPopupOpen("Export as##exportas") ? "true" : "false");
    rep_.from.clear();  // (drawn again next frame when the inspector shows it)
    const std::string j = "{\"page\":" + q(mode_name(mode_)) + ",\"keys\":" + (keys_open_ ? "true" : "false") + ",\"rows\":[" + rows + "],\"status\":" +
                          q(status_msg_) + ",\"chip\":" + b + ",\"icon\":" + rect(rep_.icon) + ",\"tabs\":" + rect(rep_.tabs) + ",\"zoom\":" + rect(rep_.zoom) +
                          ",\"popover\":" + rect(rep_.pop) + subj + ",\"windows\":[" + wins + "],\"size\":[" + std::to_string(static_cast<int>(ImGui::GetIO().DisplaySize.x)) +
                          "," + std::to_string(static_cast<int>(ImGui::GetIO().DisplaySize.y)) + "]}";
    if (j == rep_sent_) return;
    rep_sent_ = j;
    dreq("uireport", design::request("ui.report", "{\"layout\":" + j + "}"), true);
  }

  // ---------------------------------------- sub-assembly workspace ---------------------------------------------------------------
  // (state: docs / subs / exports / origins of the backend, docs/guide.md "Sub-assemblies"; every action is one sub.* backend op)
  void sub_req(const std::string& tool, const std::string& args = "{}") { dreq("sub", design::request(tool, args)); }
  std::string sub_of(const std::string& inst) const {
    const auto o = dstate_.origins.find(inst);
    return o == dstate_.origins.end() ? "" : o->second.sub;
  }
  const design::SubRow* find_sub(const std::string& id) const {
    for (const design::SubRow& s : dstate_.subs)
      if (s.id == id) return &s;
    return nullptr;
  }
  std::string fresh_sub_id(const std::string& base_name) const {
    std::string b;
    for (const char c : base_name) b += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
    if (b.empty() || std::isdigit(static_cast<unsigned char>(b[0]))) b = "p" + b;
    std::string id = b;
    for (int n = 2; find_sub(id); ++n) id = b + std::to_string(n);
    return id;
  }
  // The sub-assembly nodes of the browser tree (also what ui.report lists): one per [[subassemblies]] entry with its flattened items.
  struct SubNode {
    std::string id, file, mirror, error;
    std::vector<const design::Instance*> items;
  };
  std::vector<SubNode> sub_nodes() const {
    std::vector<SubNode> out;
    for (const design::SubRow& s : dstate_.subs) {
      SubNode n{s.id, s.file, s.mirror, "", {}};
      if (const auto e = dstate_.sub_errors.find(s.id); e != dstate_.sub_errors.end()) n.error = e->second;
      for (const design::Instance& i : design_.instances)
        if (sub_of(i.id) == s.id) n.items.push_back(&i);
      out.push_back(std::move(n));
    }
    return out;
  }

  // Return to the document at depth `to` (0 = the base design). A dirty sub-assembly document is never dropped silently: the
  // SAVE / DISCARD / CANCEL popup asks first (the backend keeps the dirty marker until it is saved or discarded).
  void close_docs(int to) {
    const int top = static_cast<int>(dstate_.docs.size()) - 1;
    if (to < 0 || to >= top) return;
    bool dirty = false;
    for (int k = to + 1; k <= top; ++k) dirty = dirty || dstate_.docs[k].dirty;
    if (!dirty) return do_close_docs(to, true, "");
    close_ask_.open = close_ask_.want_popup = true;
    close_ask_.to = to;
    const design::DocRow& d = dstate_.docs[top];
    const std::string dflt = d.path.empty() ? (d.crumb.empty() ? std::string("part") : d.crumb.back()) + ".toml" : d.path;
    std::snprintf(close_ask_.path, sizeof close_ask_.path, "%s", dflt.c_str());
  }
  void do_close_docs(int to, bool save, const std::string& path) {
    const int n = static_cast<int>(dstate_.docs.size()) - 1 - to;
    for (int k = 0; k < n; ++k)  // (built when it is its turn: a refused close, e.g. no file to save to, ends the series)
      dreq("sub", [this, to, save, path, k] {
        if (static_cast<int>(dstate_.docs.size()) - 1 <= to || (k > 0 && close_failed_)) return std::string();
        if (k == 0) close_failed_ = false;
        return design::request("sub.close", !save ? "{\"save\":false}" : k == 0 && !path.empty() ? "{\"path\":" + q(path) + "}" : "{}");
      });
  }
  void close_popup() {
    if (close_ask_.want_popup) ImGui::OpenPopup("Unsaved sub-assembly##closedoc"), close_ask_.want_popup = false;
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
    if (!ImGui::BeginPopupModal("Unsaved sub-assembly##closedoc", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
      close_ask_.open = false;
      return;
    }
    const design::DocRow* top = dstate_.docs.empty() ? nullptr : &dstate_.docs.back();
    ImGui::Text("%s has edits that are not saved.", top ? top->title.c_str() : "this part");
    ImGui::TextDisabled("Going back without saving would drop them.");
    if (top && top->path.empty()) {  // a new part: it needs a file
      ImGui::SetNextItemWidth(320);
      ImGui::InputText("##closepath", close_ask_.path, sizeof close_ask_.path);
      ImGui::SameLine();
      ImGui::TextDisabled("file (next to the design)");
    }
    ImGui::Spacing();
    if (theme::button("SAVE AND GO BACK", true) || ImGui::IsKeyPressed(ImGuiKey_Enter)) {
      do_close_docs(close_ask_.to, true, top && top->path.empty() ? close_ask_.path : "");
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("DISCARD")) {
      do_close_docs(close_ask_.to, false, "");
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("CANCEL") || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
  // "Export as...": what the part offers to the outside (an [exports] row). kind 0 port, 1 connector, 2 param.
  void export_popup() {
    if (export_ask_.want_popup) ImGui::OpenPopup("Export as##exportas"), export_ask_.want_popup = false;
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
    if (!ImGui::BeginPopupModal("Export as##exportas", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) return;
    static const char* const kinds[] = {"port", "connector", "param"};
    ImGui::SetNextItemWidth(120);
    if (ImGui::BeginCombo("kind", kinds[export_ask_.kind])) {
      for (int k = 0; k < 3; ++k)
        if (ImGui::Selectable(kinds[k], k == export_ask_.kind)) export_ask_.kind = k;
      ImGui::EndCombo();
    }
    ImGui::SetNextItemWidth(260);
    ImGui::InputText("of  (inst.name)", export_ask_.ref, sizeof export_ask_.ref);
    ImGui::SetNextItemWidth(260);
    ImGui::InputText("exported as", export_ask_.name, sizeof export_ask_.name);
    ImGui::Spacing();
    const bool ok = export_ask_.ref[0] && export_ask_.name[0];
    ImGui::BeginDisabled(!ok);
    if (theme::button("EXPORT", true) && ok) {
      sub_req("sub.export", "{\"kind\":" + q(kinds[export_ask_.kind]) + ",\"ref\":" + q(export_ask_.ref) + ",\"name\":" + q(export_ask_.name) + "}");
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("CANCEL") || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
  void ask_export(int kind, const std::string& ref, const std::string& name) {
    export_ask_.want_popup = true;
    export_ask_.kind = kind;
    std::snprintf(export_ask_.ref, sizeof export_ask_.ref, "%s", ref.c_str());
    std::snprintf(export_ask_.name, sizeof export_ask_.name, "%s", name.c_str());
  }
  // MY PARTS: insert a sub-assembly file (welded at `at`, world metres), id fresh. The file is written relative to the open design.
  void insert_my_part(const std::string& path, b3::V3 at) {
    std::string file = path;
    try {
      if (!dstate_.path.empty()) file = std::filesystem::path(path).lexically_relative(std::filesystem::path(dstate_.path).parent_path()).string();
      if (file.empty()) file = path;
    } catch (const std::exception&) {
      file = path;
    }
    const std::string id = fresh_sub_id(std::filesystem::path(path).stem().string());
    sub_req("sub.insert", "{\"file\":" + q(file) + ",\"id\":" + q(id) + ",\"pos\":" + point_json(at) + "}");
  }
  // library.my_parts reply -> the MY PARTS rows (and, when that category is open, the card list)
  void take_my_parts(const design::Json& res, bool as_hits) {
    lib_.mine.clear();
    if (as_hits) lib_.hits.clear();
    std::string needle = lib_.query;
    std::transform(needle.begin(), needle.end(), needle.begin(), [](unsigned char c) { return std::tolower(c); });
    for (const design::Json& p : res.at("parts").arr) {
      libui::MyPart m;
      m.name = p.s("name"), m.path = p.s("path");
      m.has_mass = p.at("mass").kind == design::Json::Num, m.mass = p.n("mass");
      m.has_cost = p.at("cost").kind == design::Json::Num, m.cost = p.n("cost");
      for (const auto& [kind, names] : p.at("exports").obj) m.exports += (m.exports.empty() ? "" : ", ") + std::to_string(names.arr.size()) + " " + kind;
      lib_.mine[m.path] = m;
      std::string low = m.name;
      std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return std::tolower(c); });
      if (as_hits && low.find(needle) != std::string::npos) {
        libui::Hit h{m.path, m.name, "my part", "MY PARTS", "", m.has_mass, m.mass, true};
        lib_.hits.push_back(std::move(h));
      }
    }
    lib_.my_count = static_cast<int>(lib_.mine.size());
    if (as_hits) lib_.total = static_cast<int>(lib_.hits.size());
  }

  // The left dock: BROWSER (the assembly tree: instances by group, mates, wires, devices, each instance with a visibility eye) and
  // PARTS (the component library list; cards drag into the 3D view).
  void browser_pane(float ph) {
    if (left_tab_ == 1) {
      libui::Actions a;
      libui::sidebar(lib_, *thumbs_, {0, kTop + 30}, {pane_l_, ph - 30}, a);  // (below the tab strip)
      lib_actions(a);
      ImGui::SetNextWindowPos({0, kTop});
      ImGui::SetNextWindowSize({pane_l_, 30});
      ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kPanel);
      ImGui::Begin("##ltabs", nullptr, kPane | ImGuiWindowFlags_NoScrollbar);
      ImGui::PopStyleColor();
      left_tabs();
      ImGui::Dummy({0, 0});  // (theme::tabs leaves the cursor past its last tab: an item must follow)
      ImGui::End();
      return;
    }
    begin_pane("##left", {0, kTop}, {pane_l_, ph});
    left_tabs();
    ImGui::SetCursorPos({ImGui::GetStyle().WindowPadding.x, 40});
    ImGui::BeginChild("##browser", {0, 0});
    browser_tree();
    ImGui::EndChild();
    ImGui::End();
  }
  void left_tabs() {
    static const char* const tabs[] = {"BROWSER", "PARTS"};
    const int hit = theme::tabs(tabs, 2, left_tab_, 18.f);
    if (hit >= 0) left_tab_ = hit;
  }

  // One tree row: visibility eye, label (selectable), a dim detail text at the right edge.
  bool tree_row(const std::string& id, const char* label, const std::string& detail, bool selected, bool* eye, bool hl = false, bool grey = false) {
    const float w = ImGui::GetContentRegionAvail().x, rh = ImGui::GetTextLineHeight() + 6;
    ImGui::PushID(id.c_str());
    ImGui::PushStyleColor(ImGuiCol_Text, hl ? theme::kWarn : grey ? theme::kDim : theme::kText);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (eye ? 22.f : 0.f));
    const bool hit = ImGui::Selectable(label, selected, 0, {w - (eye ? 22.f : 0.f), rh});
    ImGui::PopStyleColor();
    const ImVec2 mn = ImGui::GetItemRectMin(), mx = ImGui::GetItemRectMax();
    if (ImGui::IsItemHovered() && scene_.count(id)) tree_hover_ = id;
    if (!detail.empty()) {
      ImDrawList* dl = ImGui::GetWindowDrawList();
      ImGui::PushFont(nullptr, theme::kFontSmall);
      const float dw = ImGui::CalcTextSize(detail.c_str()).x, lw = ImGui::CalcTextSize(label).x + 18;
      if (mx.x - mn.x - lw > dw + 6) dl->AddText({mx.x - dw - 6, mn.y + 4}, theme::u32(theme::kDim), detail.c_str());
      ImGui::PopFont();
    }
    if (eye) {  // (same height as the row: the cursor then advances exactly one row)
      ImGui::SetCursorScreenPos({mn.x - 22, mn.y});
      if (theme::icon_button("##eye", *eye ? theme::Icon::Eye : theme::Icon::EyeOff, rh, false, true, *eye ? "hide this part in the 3D view" : "show this part")) *eye = !*eye;
    }
    ImGui::PopID();
    return hit;
  }

  void browser_tree() {
    using theme::Icon;
    if (!has_design_) {
      ImGui::TextDisabled("No design is open.");
      return;
    }
    ImGui::PushFont(nullptr, theme::kFontBody);
    if (ImGui::TreeNodeEx("##asm", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth, "%s", design_.assembly.c_str())) {
      std::map<std::string, std::vector<const design::Instance*>> by_group;
      for (const design::Instance& i : design_.instances)
        if (sub_of(i.id).empty()) by_group[i.group.empty() ? "(ungrouped)" : i.group].push_back(&i);
      if (ImGui::TreeNodeEx("##inst", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth, "Parts  (%zu)", design_.instances.size())) {
        for (const SubNode& sn : sub_nodes()) {  // one collapsible node per sub-assembly; its items are greyed (they live in the child file)
          const bool bad = !sn.error.empty();
          std::string lab = sn.id + "  (" + sn.file + ")" + (sn.mirror.empty() ? "" : "  [mirrored " + sn.mirror + "]") + (bad ? "  [file missing]" : "");
          ImGui::PushStyleColor(ImGuiCol_Text, bad ? theme::kErr : theme::kText);
          const bool open = ImGui::TreeNodeEx(("##sub" + sn.id).c_str(), ImGuiTreeNodeFlags_SpanAvailWidth | (sel_sub_ == sn.id ? ImGuiTreeNodeFlags_Selected : 0), "%s", lab.c_str());
          ImGui::PopStyleColor();
          if (ImGui::IsItemClicked()) sel_sub_ = sn.id;
          if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            if (bad) ImGui::TextColored(theme::kErr, "%s", sn.error.c_str());
            const auto b = design_.bom_subs.find(sn.id);
            if (b != design_.bom_subs.end()) ImGui::TextDisabled("%.1f g%s", b->second.mass * 1000, b->second.cost >= 0 ? ("   $" + std::to_string(b->second.cost).substr(0, 7)).c_str() : "");
            ImGui::TextDisabled("double-click: open the part");
            ImGui::EndTooltip();
            if (ImGui::IsMouseDoubleClicked(0) && !bad) sub_req("sub.open", "{\"id\":" + q(sn.id) + "}");
          }
          if (!open) continue;
          for (const design::Instance* i : sn.items) {
            const auto o = dstate_.origins.find(i->id);
            bool vis = !hidden_.count(i->id);
            if (tree_row(i->id, (o == dstate_.origins.end() ? i->id : o->second.id).c_str(), i->part, is_selected(i->id), &vis, hl_.count(i->id) > 0, true)) set_sel({i->id});
            if (vis != !hidden_.count(i->id)) {
              if (vis) hidden_.erase(i->id);
              else hidden_.insert(i->id);
              hidden_dirty_ = true;
              dreq("quiet", design::request("ui.visibility", "{\"instances\":[" + q(i->id) + "],\"visible\":" + (vis ? "true" : "false") + "}"), true);
            }
          }
          ImGui::TreePop();
        }
        for (const auto& [g, list] : by_group) {
          if (!ImGui::TreeNodeEx(("##g" + g).c_str(), ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth, "%s  (%zu)", g.c_str(), list.size())) continue;
          for (const design::Instance* i : list) {
            bool vis = !hidden_.count(i->id);
            const bool hl = hl_.count(i->id) > 0;
            const std::string label = i->id + (i->id == dstate_.root ? "  (root)" : "");
            if (tree_row(i->id, label.c_str(), i->part, is_selected(i->id), &vis, hl)) {
              std::vector<std::string> next = ImGui::GetIO().KeyShift || ImGui::GetIO().KeyCtrl ? sel_ : std::vector<std::string>{};
              if (!std::erase(next, i->id)) next.push_back(i->id);
              set_sel(next);
            }
            if (vis != !hidden_.count(i->id)) {
              if (vis) hidden_.erase(i->id);
              else hidden_.insert(i->id);
              hidden_dirty_ = true;
              dreq("quiet", design::request("ui.visibility", "{\"instances\":[" + q(i->id) + "],\"visible\":" + (vis ? "true" : "false") + "}"), true);  // flatten.state flags it
            }
          }
          ImGui::TreePop();
        }
        ImGui::TreePop();
      }
      const auto select_ends = [&](const std::string& a, const std::string& b) {
        set_sel({a.substr(0, a.find('.')), b.substr(0, b.find('.'))});
      };
      if (ImGui::TreeNodeEx("##mates", ImGuiTreeNodeFlags_SpanAvailWidth, "Mates  (%zu)", dstate_.mates.size())) {
        for (std::size_t k = 0; k < dstate_.mates.size(); ++k) {
          const design::Mate& m = dstate_.mates[k];
          if (tree_row("m" + std::to_string(k), (m.name.empty() ? m.a + " - " + m.b : m.name).c_str(), m.name.empty() ? "" : m.a + " - " + m.b, false, nullptr)) select_ends(m.a, m.b);
        }
        ImGui::TreePop();
      }
      if (ImGui::TreeNodeEx("##wires", ImGuiTreeNodeFlags_SpanAvailWidth, "Wires  (%zu)", dstate_.wires.size())) {
        for (std::size_t k = 0; k < dstate_.wires.size(); ++k) {
          const design::WireRow& w = dstate_.wires[k];
          if (tree_row("w" + std::to_string(k), (w.name.empty() ? w.from + " > " + w.to : w.name).c_str(), w.name.empty() ? "" : w.from + " > " + w.to, false, nullptr)) select_ends(w.from, w.to);
        }
        ImGui::TreePop();
      }
      if (!dstate_.exports.empty() && ImGui::TreeNodeEx("##exports", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth, "Exports  (%zu)", dstate_.exports.size())) {
        for (std::size_t k = 0; k < dstate_.exports.size(); ++k) {  // what this part offers to a parent: x removes the row
          const design::ExportRow& e = dstate_.exports[k];
          const float y0 = ImGui::GetCursorPosY(), w = ImGui::GetContentRegionAvail().x;
          if (tree_row("x" + std::to_string(k), e.name.c_str(), e.kind + "  " + e.ref, false, nullptr)) set_sel({e.ref.substr(0, e.ref.find_first_of(".:"))});
          const float y1 = ImGui::GetCursorPosY();
          ImGui::SetCursorPos({ImGui::GetCursorPosX() + w - 22, y0});
          ImGui::PushID(static_cast<int>(k));
          if (ImGui::SmallButton("x")) sub_req("sub.unexport", "{\"kind\":" + q(e.kind == "ports" ? "port" : e.kind == "connectors" ? "connector" : "param") + ",\"name\":" + q(e.name) + "}");
          ImGui::PopID();
          ImGui::SetItemTooltip("remove this export");
          ImGui::SetCursorPosY(y1);
        }
        ImGui::TreePop();
      }
      if (ImGui::TreeNodeEx("##devs", ImGuiTreeNodeFlags_SpanAvailWidth, "Devices  (%zu)", dstate_.devices.size())) {
        for (std::size_t k = 0; k < dstate_.devices.size(); ++k) {
          const design::Device& d = dstate_.devices[k];
          tree_row("d" + std::to_string(k), d.name.c_str(), d.type, false, nullptr);
        }
        ImGui::TreePop();
      }
      ImGui::TreePop();
    }
    ImGui::PopFont();
  }

  void inspector_pane(float W, float ph) {
    begin_pane("##right", pane_pos(true, W), {pane_r_, ph});
    static const char* const tabs[] = {"PROPERTIES", "DRC", "PLACE", "HISTORY"};
    static const int tab_ids[] = {0, 2, 3, 4};  // right_tab_ values (1 = the old PARTS list, now the BROWSER on the left)
    int cur = 0;
    for (int k = 0; k < 4; ++k)
      if (tab_ids[k] == right_tab_) cur = k;
    const int hit = theme::tabs(tabs, 4, cur, 16.f);
    if (hit >= 0) right_tab_ = tab_ids[hit];
    ImGui::SetCursorPos({ImGui::GetStyle().WindowPadding.x, ImGui::GetCursorPosY() + 30});  // theme::tabs leaves the cursor on its row
    ImGui::BeginChild("##rbody", {0, 0});
    const design::Instance* sel = sel_instance();
    if (right_tab_ == 0) inspect_tab(sel);
    else if (right_tab_ == 1) instance_list();
    else if (right_tab_ == 2) drc_body(true);
    else if (right_tab_ == 4) history_tab();
    else {
      if (place_.alts.empty()) ImGui::TextDisabled("Snap a part to see its placement alternatives here.");
      placement_ui();
    }
    ImGui::EndChild();
    ImGui::End();
  }

  // Undo history (python Design._undo_notes): the steps behind the document, newest last, then the steps UNDO took back. A click
  // on a row travels there (one undo / redo request per step).
  void history_tab() {
    theme::caption("EDIT HISTORY");
    if (dstate_.hist_undo.empty() && dstate_.hist_redo.empty()) {
      ImGui::Spacing();
      ImGui::TextDisabled("No edits yet. Add, move or wire a part and each step is listed here.");
      ImGui::TextDisabled("Ctrl+Z undoes the last one, Ctrl+Shift+Z brings it back.");
      return;
    }
    const int n = static_cast<int>(dstate_.hist_undo.size());
    const int total = dstate_.undo;  // steps older than the list (the backend keeps the last 14)
    if (total > n) ImGui::TextDisabled("... %d earlier steps", total - n);
    for (int i = 0; i < n; ++i) {
      const bool last = i == n - 1;
      ImGui::PushID(i);
      ImGui::PushStyleColor(ImGuiCol_Text, last ? theme::kText : theme::kDim);
      const std::string row = std::to_string(total - n + i + 1) + "   " + dstate_.hist_undo[i];
      if (ImGui::Selectable(row.c_str(), last)) {
        for (int k = 0; k < n - 1 - i; ++k) d_undo(false);  // back to just after this step
      }
      ImGui::PopStyleColor();
      ImGui::PopID();
    }
    if (!dstate_.hist_redo.empty()) {
      ImGui::Spacing();
      theme::caption("UNDONE (CTRL+SHIFT+Z)");
      for (std::size_t i = 0; i < dstate_.hist_redo.size(); ++i) {
        ImGui::PushID(1000 + static_cast<int>(i));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{theme::kDim.x, theme::kDim.y, theme::kDim.z, 0.6f});
        if (ImGui::Selectable(dstate_.hist_redo[i].c_str(), false))
          for (std::size_t k = 0; k <= i; ++k) d_undo(true);
        ImGui::PopStyleColor();
        ImGui::PopID();
      }
    }
  }

  // The primary selection's pose in the document's terms: world position (mm) and rpy (deg).
  void inspect_tab(const design::Instance* sel) {
    if (!opts_.design_api && !has_design_) return;
    ImGui::TextDisabled("%s   total %.4f kg   %zu parts", design_.assembly.c_str(), design_.mass_total, design_.instances.size());
    ImGui::Spacing();
    if (!sel) {
      ImGui::TextDisabled("click a part in the 3D view or the browser");
      sub_panel();
      snapping_controls();
      return;
    }
    ImGui::PushFont(nullptr, 17.f);
    ImGui::TextUnformatted(sel->id.c_str());
    ImGui::PopFont();
    ImGui::TextDisabled("%s", sel->part.c_str());
    ImGui::TextDisabled("%s   %.1f g", sel->group.empty() ? "(no group)" : sel->group.c_str(), sel->mass * 1000);
    if (const auto o = dstate_.origins.find(sel->id); o != dstate_.origins.end()) {  // an item of a sub-assembly: read-only here
      ImGui::Spacing();
      rep_.from = "from " + o->second.file;
      ImGui::TextColored(theme::kAccent, "%s", rep_.from.c_str());
      ImGui::TextDisabled("%s of sub-assembly %s", o->second.id.c_str(), o->second.sub.c_str());
      ImGui::TextDisabled("read-only here: edit it in the part.");
      const auto err = dstate_.sub_errors.find(o->second.sub);
      if (err != dstate_.sub_errors.end()) ImGui::TextColored(theme::kErr, "%s", err->second.c_str());
      ImGui::Spacing();
      ImGui::BeginDisabled(err != dstate_.sub_errors.end());
      if (theme::button("OPEN PART", true)) sub_req("sub.open", "{\"id\":" + q(o->second.sub) + "}");
      ImGui::EndDisabled();
      ImGui::SameLine();
      if (ImGui::Button("FRAME")) frame_selection();
      return;
    }
    const bool root = sel->id == dstate_.root;
    const bool is_free = std::find(dstate_.free.begin(), dstate_.free.end(), sel->id) != dstate_.free.end();
    ImGui::TextColored(root ? theme::kAccent : is_free ? theme::kWarn : theme::kDim, "%s", root ? "ROOT" : is_free ? "FREE-PLACED" : "MATED");
    if (sel_.size() > 1) {
      ImGui::SameLine();
      ImGui::TextDisabled("  +%zu more selected", sel_.size() - 1);
    }
    const auto git = scene_.find(sel->id);
    if (git != scene_.end() && !root && opts_.design_api) {
      const b3::InstGeo& g = git->second;
      ImGui::Spacing();
      theme::caption("POSITION  MM");
      double p[3] = {g.pos.x * 1000, g.pos.y * 1000, g.pos.z * 1000};
      bool commit = false;
      ImGui::PushID("pos");
      for (int k = 0; k < 3; ++k) {
        ImGui::PushID(k);
        ImGui::SetNextItemWidth((pane_r_ - 24 - 16) / 3);
        ImGui::DragScalar("##v", ImGuiDataType_Double, &p[k], 0.5f, nullptr, nullptr, "%.1f");
        commit = commit || ImGui::IsItemDeactivatedAfterEdit();
        ImGui::PopID();
        if (k < 2) ImGui::SameLine();
      }
      ImGui::PopID();
      if (commit) place_world(sel->id, "pos", {p[0] / 1000, p[1] / 1000, p[2] / 1000});
      theme::caption("ROTATION  DEG  (R P Y)");
      const b3::V3 rpy = b3::mat_to_rpy(g.R);
      double r[3] = {rpy.x * 180 / 3.14159265358979, rpy.y * 180 / 3.14159265358979, rpy.z * 180 / 3.14159265358979};
      bool rc = false;
      ImGui::PushID("rot");
      for (int k = 0; k < 3; ++k) {
        ImGui::PushID(k);
        ImGui::SetNextItemWidth((pane_r_ - 24 - 16) / 3);
        ImGui::DragScalar("##v", ImGuiDataType_Double, &r[k], 0.5f, nullptr, nullptr, "%.1f");
        rc = rc || ImGui::IsItemDeactivatedAfterEdit();
        ImGui::PopID();
        if (k < 2) ImGui::SameLine();
      }
      ImGui::PopID();
      if (rc) place_world(sel->id, "rpy", {r[0] * 3.14159265358979 / 180, r[1] * 3.14159265358979 / 180, r[2] * 3.14159265358979 / 180});
      ImGui::Spacing();
      theme::caption("COLOUR");
      float col[4] = {0.6f, 0.6f, 0.65f, 1};
      if (g.has_color) std::copy(g.color, g.color + 4, col);
      else if (const auto it = color_pick_.find(sel->id); it != color_pick_.end()) std::copy(it->second.begin(), it->second.end(), col);
      if (ImGui::ColorButton("##swatch", {col[0], col[1], col[2], col[3]}, ImGuiColorEditFlags_AlphaPreviewHalf, {pane_r_ - 24, 22})) ImGui::OpenPopup("##colour");
      if (color_flash_ > ImGui::GetTime() && !ImGui::IsPopupOpen("##colour")) ImGui::OpenPopup("##colour");  // a scripted ui.* colour op shows the picker
      if (ImGui::BeginPopup("##colour")) {
        std::array<float, 4>& pick = color_pick_[sel->id];
        if (pick == std::array<float, 4>{}) std::copy(col, col + 4, pick.begin());
        if (ImGui::ColorPicker4("##picker", pick.data(), ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_DisplayRGB)) color_dirty_ = true;
        if (color_dirty_ && !ImGui::IsMouseDown(0) && color_flash_ <= ImGui::GetTime()) {  // released: one undo step
          color_dirty_ = false;
          dreq("edit", design::request("set_color", "{\"silent\":true,\"instance\":" + q(sel->id) + ",\"rgba\":[" + std::to_string(pick[0]) + "," + std::to_string(pick[1]) + "," + std::to_string(pick[2]) + "," + std::to_string(pick[3]) + "]}"));
        }
        if (ImGui::SmallButton("RESET TO PART COLOUR")) {
          color_pick_.erase(sel->id);
          dreq("edit", design::request("set_color", "{\"silent\":true,\"instance\":" + q(sel->id) + ",\"rgba\":null}"));
          ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
      }
    }
    if (dstate_.docs.size() > 1) {  // editing a part: what it offers to a parent
      ImGui::Spacing();
      theme::caption("EXPORT TO A PARENT");
      if (const auto pit = dstate_.ports.find(sel->id); pit != dstate_.ports.end())
        for (const design::Port& p : pit->second) {
          ImGui::PushID(p.name.c_str());
          ImGui::TextUnformatted(p.name.c_str());
          ImGui::SameLine(120);
          ImGui::TextDisabled("%s", p.kind.c_str());
          ImGui::SameLine(std::max(170.f, pane_r_ - 110));
          if (ImGui::SmallButton("Export as...")) ask_export(p.kind == "mech" || p.kind.empty() ? 0 : 1, sel->id + "." + p.name, p.name);
          ImGui::PopID();
        }
      if (ImGui::SmallButton("Export a param as...")) ask_export(2, sel->id + ".", "");
    }
    ImGui::Spacing();
    theme::caption("ACTIONS");
    const bool can = opts_.design_api && !root;
    ImGui::BeginDisabled(!can);
    if (ImGui::Button("DUPLICATE") && can) dreq("edit", design::request("duplicate", "{\"silent\":true,\"instance\":" + q(sel->id) + "}"));
    ImGui::SameLine();
    static int mirror_axis = 1;
    if (ImGui::Button("MIRROR")) dreq("edit", design::request("mirror", "{\"silent\":true,\"instances\":" + json_list(sel_) + ",\"axis\":\"" + "xyz"[mirror_axis] + "\"}"));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(46);
    const char* const axes[] = {"X", "Y", "Z"};
    if (ImGui::BeginCombo("##maxis", axes[mirror_axis])) {
      for (int k = 0; k < 3; ++k)
        if (ImGui::Selectable(axes[k], k == mirror_axis)) mirror_axis = k;
      ImGui::EndCombo();
    }
    ImGui::SetItemTooltip("mirror across the world plane through the origin normal to this axis (free-placed copies, mates are not mirrored)");
    if (ImGui::Button("REMOVE") && can) d_remove(sel->id);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("FRAME")) frame_selection();
    if (sel_.size() > 1) {
      ImGui::Spacing();
      theme::caption("ALIGN / DISTRIBUTE");
      const char* const to[] = {"min", "center", "max"};
      for (int ax = 0; ax < 3; ++ax) {
        ImGui::TextUnformatted(axes[ax]);
        for (int t = 0; t < 3; ++t) {
          ImGui::SameLine();
          ImGui::PushID(ax * 3 + t);
          if (ImGui::SmallButton(to[t])) place_group("align", ax, to[t]);
          ImGui::PopID();
        }
        ImGui::SameLine();
        ImGui::PushID(100 + ax);
        if (ImGui::SmallButton("distribute")) place_group("distribute", ax, "");
        ImGui::PopID();
      }
    }
    if (!edit_lines_.empty()) {
      ImGui::Spacing();
      theme::caption("LAST EDIT");
      ImGui::PushTextWrapPos();
      for (std::size_t i = 0; i < edit_lines_.size() && i < 6; ++i) ImGui::TextDisabled("%s", edit_lines_[i].c_str());
      ImGui::PopTextWrapPos();
    }
    snapping_controls();
  }

  // Per sub-assembly mass and cost (report["bom"].by_sub), then the total. A cost with "+" is a lower bound (parts without a price).
  void sub_panel() {
    rep_.panel.clear();
    if (dstate_.subs.empty()) return;
    ImGui::Spacing();
    theme::caption("SUB-ASSEMBLIES  MASS / COST");
    const auto cost_text = [](double c, int missing) {
      if (c < 0) return std::string("n/a");
      char b[32];
      std::snprintf(b, sizeof b, "$%.2f%s", c, missing > 0 ? "+" : "");
      return std::string(b);
    };
    for (const SubNode& sn : sub_nodes()) {
      const auto b = design_.bom_subs.find(sn.id);
      char m[32] = "-";
      if (b != design_.bom_subs.end()) std::snprintf(m, sizeof m, "%.1f g", b->second.mass * 1000);
      const std::string c = b != design_.bom_subs.end() ? cost_text(b->second.cost, b->second.missing) : "n/a";
      ImGui::TextUnformatted(sn.id.c_str());
      ImGui::SameLine(110);
      ImGui::TextDisabled("%s", m);
      ImGui::SameLine(180);
      ImGui::TextDisabled("%s", c.c_str());
      rep_.panel.push_back(sn.id + " | " + m + " | " + c);
    }
    if (design_.has_bom) {
      const std::string c = cost_text(design_.bom_cost, design_.bom_missing);
      ImGui::TextUnformatted("total");
      ImGui::SameLine(110);
      ImGui::TextDisabled("%.1f g", design_.bom_mass * 1000);
      ImGui::SameLine(180);
      ImGui::TextDisabled("%s", c.c_str());
      if (design_.bom_missing > 0) ImGui::TextDisabled("%d part(s) without a price: cost is a lower bound", design_.bom_missing);
    }
  }

  void snapping_controls() {
    ImGui::Spacing();
    theme::caption("SNAPPING");
    static const double grids[] = {0, 0.001, 0.005, 0.01, 0.05};
    static const char* const gl[] = {"off", "1 mm", "5 mm", "10 mm", "50 mm"};
    int gi = 2;
    for (int k = 0; k < 5; ++k)
      if (std::abs(snap_.grid - grids[k]) < 1e-9) gi = k;
    ImGui::SetNextItemWidth(88);
    if (ImGui::BeginCombo("grid", gl[gi])) {
      for (int k = 0; k < 5; ++k)
        if (ImGui::Selectable(gl[k], k == gi)) snap_.grid = grids[k];
      ImGui::EndCombo();
    }
    ImGui::SameLine();
    static const double angs[] = {0, 5, 15, 45, 90};
    static const char* const al[] = {"off", "5 deg", "15 deg", "45 deg", "90 deg"};
    int ai = 2;
    for (int k = 0; k < 5; ++k)
      if (std::abs(snap_.angle - angs[k]) < 1e-9) ai = k;
    ImGui::SetNextItemWidth(88);
    if (ImGui::BeginCombo("angle", al[ai])) {
      for (int k = 0; k < 5; ++k)
        if (ImGui::Selectable(al[k], k == ai)) snap_.angle = angs[k];
      ImGui::EndCombo();
    }
    ImGui::Checkbox("edges", &snap_.edges);
    ImGui::SameLine();
    ImGui::Checkbox("ports", &snap_.ports);
    ImGui::SameLine();
    ImGui::Checkbox("guides", &snap_.guides);
    ImGui::SetItemTooltip("hold Ctrl while dragging to turn all snapping off for that drag");
  }

  // A numeric / group edit of the selection in world terms (mm fields, align): confirm first when it would detach a mated part.
  void place_world(const std::string& id, const char* key, b3::V3 v) {
    const std::string tool = std::string(key) == "pos" ? "gizmo.move" : "gizmo.rotate";
    char b[160];
    std::snprintf(b, sizeof b, "[%.6f,%.6f,%.6f]", v.x, v.y, v.z);
    const std::string vec = b;
    run_placement(id, [tool, id, key, vec](bool unmate) {
      return design::request(tool, "{\"silent\":true,\"instance\":" + design::quote(id) + ",\"" + key + "\":" + vec + (unmate ? ",\"unmate\":true" : "") + "}");
    });
  }
  void place_group(const char* tool, int ax, const std::string& to) {
    const std::string ids = json_list(sel_);
    bool held = false;
    for (const std::string& i : sel_) held = held || is_held(i);
    auto make = [tool = std::string(tool), ax, to, ids](bool unmate) {
      return design::request(tool, "{\"silent\":true,\"instances\":" + ids + ",\"axis\":\"" + "xyz"[ax] + "\",\"to\":\"" + to + "\"" + (unmate ? ",\"unmate\":true" : "") + "}");
    };
    if (held) confirm_unmate(sel_.back(), make);
    else dreq("edit", make(false));
  }
  bool is_held(const std::string& id) const {
    return id != dstate_.root && std::find(dstate_.free.begin(), dstate_.free.end(), id) == dstate_.free.end();
  }
  void run_placement(const std::string& id, std::function<std::string(bool)> make) {
    if (is_held(id)) confirm_unmate(id, std::move(make));
    else dreq("edit", make(false));
  }
  void confirm_unmate(const std::string& id, std::function<std::string(bool)> make) {
    confirm_ = std::move(make);
    confirm_id_ = id;
    confirm_open_ = true;
  }
  void confirm_popup() {
    if (confirm_open_) {
      ImGui::OpenPopup("DETACH FROM MATE?");
      confirm_open_ = false;
    }
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
    if (ImGui::BeginPopupModal("DETACH FROM MATE?", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
      ImGui::Text("%s is held by a mate. Placing it freely drops that mate\n(its fasteners go with it; parts mated to it stay on it).", confirm_id_.c_str());
      ImGui::Spacing();
      if (theme::button("DETACH AND PLACE", true) || ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        dreq("edit", confirm_(true));
        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      if (ImGui::Button("CANCEL") || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
      ImGui::EndPopup();
    }
  }

  // Viewport furniture (BUILD, top-right of the 3D view): the view cube (click a face for that standard view, drag nothing), the
  // navigation buttons, and bottom-left a scale bar with the camera distance. Nothing here overlaps the docked panels.
  void viewport_overlay() {
    using theme::Icon;
    constexpr float kBoxW = 144, kCube = 26;
    const float bx = vp_.z - kBoxW - 8, by = vp_.y + 8;
    ImGui::SetNextWindowPos({bx, by});
    ImGui::SetNextWindowSize({kBoxW, 150});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4{0, 0, 0, 0});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::Begin("##vpnav", nullptr, kPane | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const b3::Cam cam = b3::Cam::make(view_cam_, ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y);
    const ImVec2 c{bx + kBoxW / 2, by + 62};
    ImGui::SetCursorScreenPos({bx, by});
    ImGui::InvisibleButton("##cube", {kBoxW, 118});  // (captures the pointer over the cube so it does not orbit the scene)
    const ImVec2 mp = ImGui::GetIO().MousePos;
    const auto proj = [&](b3::V3 v) { return ImVec2{c.x + static_cast<float>(b3::dot(v, cam.r)) * kCube, c.y - static_cast<float>(b3::dot(v, cam.u)) * kCube}; };
    struct Face {
      const char* name;
      b3::V3 n;
      double yaw, pitch;
    };
    static const Face faces[] = {{"RIGHT", {1, 0, 0}, 0, 0}, {"LEFT", {-1, 0, 0}, 3.14159265, 0}, {"BACK", {0, 1, 0}, 1.5707963, 0},
                                 {"FRONT", {0, -1, 0}, -1.5707963, 0}, {"TOP", {0, 0, 1}, -1.5707963, 1.5}, {"BOTTOM", {0, 0, -1}, -1.5707963, -1.5}};
    // back to front: the faces most facing away first (only the visible ones are drawn, so painter's order is by facing)
    int order[6] = {0, 1, 2, 3, 4, 5};
    std::sort(order, order + 6, [&](int a, int b) { return b3::dot(faces[a].n, cam.f) > b3::dot(faces[b].n, cam.f); });
    int hit = -1;
    for (int oi = 0; oi < 6; ++oi) {
      const Face& f = faces[order[oi]];
      const double facing = -b3::dot(f.n, cam.f);
      if (facing <= 0.02) continue;
      // the face's two in-plane axes
      const b3::V3 u = std::abs(f.n.z) > 0.5 ? b3::V3{1, 0, 0} : b3::V3{0, 0, 1}, v = b3::cross(f.n, u);
      const ImVec2 q[4] = {proj(f.n + u + v), proj(f.n - u + v), proj(f.n - u - v), proj(f.n + u - v)};
      const auto inside = [&] {  // convex quad: all cross products one sign
        int pos = 0, neg = 0;
        for (int k = 0; k < 4; ++k) {
          const ImVec2 a = q[k], b = q[(k + 1) % 4];
          ((b.x - a.x) * (mp.y - a.y) - (b.y - a.y) * (mp.x - a.x) >= 0 ? pos : neg)++;
        }
        return pos == 4 || neg == 4;
      }();
      const bool hov = inside && ImGui::IsItemHovered();
      if (hov) hit = order[oi];
      const float shade = 0.20f + 0.16f * static_cast<float>(facing);
      dl->AddConvexPolyFilled(q, 4, hov ? theme::u32(theme::kAccent, 0.9f) : theme::u32({shade + 0.04f, shade + 0.05f, shade + 0.08f, 1.f}, 0.95f));
      dl->AddPolyline(q, 4, theme::u32({1, 1, 1, 0.45f}), ImDrawFlags_Closed, 1.f);
      if (facing > 0.3) {
        ImGui::PushFont(nullptr, 9.5f + 1.5f * static_cast<float>(facing));
        const ImVec2 ts = ImGui::CalcTextSize(f.name), mid{(q[0].x + q[2].x) / 2, (q[0].y + q[2].y) / 2};
        dl->AddText({mid.x - ts.x / 2, mid.y - ts.y / 2}, theme::u32(hov ? ImVec4{1, 1, 1, 1} : theme::kText, static_cast<float>(std::min(1.0, facing * 1.6))), f.name);
        ImGui::PopFont();
      }
    }
    if (hit >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) camera_to(faces[hit].yaw, faces[hit].pitch);
    if (hit >= 0) ImGui::SetTooltip("%s view", faces[hit].name);
    // axis triad from the cube's lower-left corner region: X red, Y green, Z blue
    const char* const an[3] = {"X", "Y", "Z"};
    const ImVec2 o{bx + 14, by + 100};
    for (int a = 0; a < 3; ++a) {
      const b3::V3 ax = b3::axis_vec(a);
      const ImVec2 e{o.x + static_cast<float>(b3::dot(ax, cam.r)) * 16, o.y - static_cast<float>(b3::dot(ax, cam.u)) * 16};
      dl->AddLine(o, e, b3::axis_color(a), 2.f);
      ImGui::PushFont(nullptr, 9.5f);
      dl->AddText({e.x + (e.x >= o.x ? 2 : -8), e.y - 5}, b3::axis_color(a), an[a]);
      ImGui::PopFont();
    }
    // navigation buttons
    float x = bx + kBoxW - 5 * 28;
    ImGui::SetCursorScreenPos({x, by + 120});
    ImGui::PushStyleColor(ImGuiCol_Text, theme::kText);
    if (theme::icon_button("##home", Icon::Home, 28, false, true, "isometric view")) camera_preset("iso");
    ImGui::SetCursorScreenPos({x + 28, by + 120});
    if (theme::icon_button("##fit", Icon::Fit, 28, false, true, "zoom to fit the selection (or everything); F")) frame_selection();
    ImGui::SetCursorScreenPos({x + 56, by + 120});
    if (theme::icon_button("##orbit", Icon::Orbit, 28, !nav_pan_, true, "orbit: left-drag turns the view (wheel zooms)")) nav_pan_ = false;
    ImGui::SetCursorScreenPos({x + 84, by + 120});
    if (theme::icon_button("##pan", Icon::Pan, 28, nav_pan_, true, "pan: left-drag slides the view (right/middle drag always pans)")) nav_pan_ = true;
    ImGui::SetCursorScreenPos({x + 112, by + 120});
    if (theme::icon_button("##wires", Icon::Net, 28, emph_wires_, true, "emphasise cables: at least 4 px wide at any zoom (true diameter unchanged)")) emph_wires_ = !emph_wires_;
    ImGui::PopStyleColor();
    ImGui::End();
    scale_bar();
  }

  // Bottom-left of the viewport: a scale bar (rounded to 1/2/5 x 10^n) at the orbit target's depth and the camera distance.
  void scale_bar() {
    const ImGuiIO& io = ImGui::GetIO();
    const b3::Cam cam = b3::Cam::make(view_cam_, io.DisplaySize.x, io.DisplaySize.y);
    const b3::V3 t{orbit_.target[0], orbit_.target[1], orbit_.target[2]};
    const double ppm = cam.px_per_m(t);
    if (!(ppm > 0)) return;
    double m = 1e-3;
    for (const double s : {1.0, 2.0, 5.0})
      for (double e = 1e-4; e < 1e3; e *= 10)
        if (s * e * ppm <= 140 && s * e > m) m = s * e;
    const float len = static_cast<float>(m * ppm);
    char lab[48];
    if (m < 1) std::snprintf(lab, sizeof lab, "%g mm", m * 1000);
    else std::snprintf(lab, sizeof lab, "%g m", m);
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const float x0 = vp_.x + 16, y0 = vp_.w - 18;
    if (y0 < vp_.y + 60) return;
    const ImU32 col = IM_COL32(40, 46, 58, 230), halo = IM_COL32(255, 255, 255, 140);
    for (int pass = 0; pass < 2; ++pass) {
      const float th = pass == 0 ? 4.f : 1.6f;
      const ImU32 cc = pass == 0 ? halo : col;
      dl->AddLine({x0, y0}, {x0 + len, y0}, cc, th);
      dl->AddLine({x0, y0 - 4}, {x0, y0 + 4}, cc, th);
      dl->AddLine({x0 + len, y0 - 4}, {x0 + len, y0 + 4}, cc, th);
    }
    ImGui::PushFont(nullptr, theme::kFontSmall);
    dl->AddText({x0 + 1, y0 - 19}, halo, lab);
    dl->AddText({x0, y0 - 20}, col, lab);
    ImGui::PopFont();
  }

  // The poses handed to the renderer: the simulation's, except that geoms of instances hidden in the browser are collapsed to a point
  // (zero rotation = zero area). A geom belongs to the smallest instance box that contains its origin (the visuals carry no body
  // link); geoms in no box (floor, loose sim geoms) stay. Approximate by design, exact for parts that do not overlap.
  // The instance a point belongs to: the smallest box that contains it; else (a mesh origin a few mm off its part's box, e.g. a propeller
  // hub 5 mm above the 4 mm thick prop box) the nearest box within `slack` metres; else "" (floor, loose geoms).
  std::string owner_at(const double* p, double slack) const {
    std::string own;
    double best = 1e300;
    for (const auto& [id, s] : scene_) {
      bool in = true;
      double vol = 1;
      for (int k = 0; k < 3; ++k) {
        in = in && p[k] >= s.lo[k] - 1e-4 && p[k] <= s.hi[k] + 1e-4;
        vol *= s.hi[k] - s.lo[k] + 2e-4;
      }
      if (in && vol < best) best = vol, own = id;
    }
    if (!own.empty()) return own;
    double dmin = slack;
    for (const auto& [id, s] : scene_) {
      double d2 = 0;
      for (int k = 0; k < 3; ++k) {
        const double e = std::max({s.lo[k] - p[k], p[k] - s.hi[k], 0.0});
        d2 += e * e;
      }
      if (d2 <= dmin * dmin) dmin = std::sqrt(d2), own = id;
    }
    return own;
  }

  // The cables handed to the renderer: those with an end on a hidden instance are dropped. The ends are the instance ids the design
  // builder put in the tendon's name (Frame::tube_ends), not a guess from the geometry.
  std::span<const TubeVisual> shown_tubes() {
    if (hidden_.empty()) return cur_.tubes;
    shown_tubes_.clear();
    for (std::size_t i = 0; i < cur_.tubes.size(); ++i) {
      const bool ends = i < cur_.tube_ends.size();
      if (!ends || (!hidden_.count(cur_.tube_ends[i].first) && !hidden_.count(cur_.tube_ends[i].second))) shown_tubes_.push_back(cur_.tubes[i]);
    }
    return shown_tubes_;
  }

  const std::vector<GeomPose>& shown_poses() {
    if (hidden_.empty()) return cur_.poses;
    const std::string key = dstate_.rev + "/" + std::to_string(frame_scene_) + "/" + std::to_string(cur_.poses.size());
    if (key != geom_owner_key_) {
      geom_owner_key_ = key;
      geom_owner_.assign(cur_.poses.size(), "");
      for (std::size_t g = 0; g < cur_.poses.size(); ++g) geom_owner_[g] = owner_at(cur_.poses[g].pos, 0.03);
    }
    hidden_poses_ = cur_.poses;
    for (std::size_t g = 0; g < hidden_poses_.size() && g < geom_owner_.size(); ++g)
      if (!geom_owner_[g].empty() && hidden_.count(geom_owner_[g])) std::fill(std::begin(hidden_poses_[g].rot), std::end(hidden_poses_[g].rot), 0.0);
    return hidden_poses_;
  }

  // camera.orbit / camera.fly (python/dvsim/design/uiops.py): a smooth move over `seconds` of wall-clock time.
  void camera_move(const design::Json& e) {
    if (e.at("chase").num != 0 && e.at("chase").kind == design::Json::Bool && !run_.follow) return;  // the user holds the camera
    constexpr double kDeg = 3.14159265358979 / 180;
    const auto vec3 = [&](const char* k, b3::V3& out) {
      double p[3];
      if (e.at(k).kind != design::Json::Arr || e.at(k).arr.size() < 3) return false;
      design::vec_from(e.at(k), p, 3);
      out = {p[0], p[1], p[2]};
      return true;
    };
    CamAnim a;
    a.active = true;
    a.t0 = ImGui::GetTime();
    a.dur = std::max(0.0, e.n("seconds", 3.0));
    a.linear = e.s("ease") == "linear";
    a.yaw0 = a.yaw1 = orbit_.yaw, a.pitch0 = a.pitch1 = orbit_.pitch, a.dist0 = a.dist1 = orbit_.dist;
    const Camera now_cam = orbit_.camera();
    if (e.s("kind") == "fly") {
      a.fly = true;
      a.eye0 = {now_cam.pos[0], now_cam.pos[1], now_cam.pos[2]};
      a.look0 = {orbit_.target[0], orbit_.target[1], orbit_.target[2]};
      if (!vec3("eye", a.eye1) || !vec3("look", a.look1)) return;
    } else {
      const bool rel = e.at("relative").kind != design::Json::Bool || e.at("relative").num != 0;
      a.sweep = rel;
      if (vec3("target", a.target_to)) {
        a.move_target = true;
        a.target_from = {orbit_.target[0], orbit_.target[1], orbit_.target[2]};
      }
      if (e.at("yaw_deg").kind == design::Json::Num) a.yaw1 = rel ? orbit_.yaw + e.n("yaw_deg") * kDeg : e.n("yaw_deg") * kDeg;
      if (e.at("pitch_deg").kind == design::Json::Num) a.pitch1 = std::clamp(rel ? orbit_.pitch + e.n("pitch_deg") * kDeg : e.n("pitch_deg") * kDeg, -1.5, 1.5);
      if (e.at("dist").kind == design::Json::Num) a.dist1 = std::max(0.02, e.n("dist"));
    }
    cam_anim_ = a;
    if (a.dur <= 1e-3) step_cam_anim(a.t0);  // (instant)
  }

  void camera_to(double yaw, double pitch) {
    cam_anim_ = {true, ImGui::GetTime(), orbit_.yaw, orbit_.pitch, orbit_.dist, yaw, pitch, orbit_.dist};
  }
  void camera_preset(const std::string& p) {
    static const std::map<std::string, std::pair<double, double>> pre = {{"front", {-1.5707963, 0.0}}, {"side", {0.0, 0.0}}, {"top", {-1.5707963, 1.5}}, {"iso", {-0.6, 0.55}},
                                                                         {"back", {1.5707963, 0.0}}, {"left", {3.14159265, 0.0}}, {"bottom", {-1.5707963, -1.5}}};
    const auto it = pre.find(p);
    if (it == pre.end()) return;
    cam_anim_ = {true, ImGui::GetTime(), orbit_.yaw, orbit_.pitch, orbit_.dist, it->second.first, it->second.second, orbit_.dist};
  }
  // Frame the selection (everything when nothing is selected or `all`); `instant`: no glide (the first frame of a design).
  void frame_selection(bool all = false, bool instant = false) {
    b3::V3 lo{1e9, 1e9, 1e9}, hi{-1e9, -1e9, -1e9};
    bool any = false;
    for (const std::string& id : sel_) {
      if (all) break;
      const auto it = scene_.find(id);
      if (it == scene_.end()) continue;
      for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], it->second.lo[k]);
        hi[k] = std::max(hi[k], it->second.hi[k]);
      }
      any = true;
    }
    if (!any) {
      for (const auto& [id, g] : scene_)
        for (int k = 0; k < 3; ++k) {
          lo[k] = std::min(lo[k], g.lo[k]);
          hi[k] = std::max(hi[k], g.hi[k]);
          any = true;
        }
    }
    if (!any) return;
    const b3::V3 c = (lo + hi) * 0.5;
    const double r = std::max(0.03, 0.5 * b3::len(hi - lo));
    cam_anim_ = {true, ImGui::GetTime(), orbit_.yaw, orbit_.pitch, orbit_.dist, orbit_.yaw, orbit_.pitch, 2.6 * r};
    cam_anim_.target_from = {orbit_.target[0], orbit_.target[1], orbit_.target[2]};
    cam_anim_.target_to = c;
    cam_anim_.move_target = true;
    if (instant) cam_anim_.dur = 0, step_cam_anim(cam_anim_.t0);
  }

  // The sidebar / page asked for something.
  void lib_actions(const libui::Actions& a) {
    if (!a.info.empty() && !lib_.mine.count(a.info)) want_info(a.info);
    if (!a.open_page.empty()) {
      lib_.selected = a.open_page;
      set_mode("library");
      if (!lib_.mine.count(a.open_page)) want_info(a.open_page);
    }
    if (!a.add_mine.empty()) insert_my_part(a.add_mine, free_drop_point());
    if (!a.add.empty()) add_part(a.add);
    if (!a.swap.empty()) {
      if (const design::Instance* s = sel_instance()) d_swap(s->id, a.swap, true);
    }
  }
  void want_info(const std::string& id) {
    if (opts_.design_api && !lib_.infos.count(id)) {
      lib_.infos[id].id = "";  // (marks the request as sent)
      dreq("info", design::request("library.info", "{\"part\":" + q(id) + "}"), true);
    }
  }
  // ADD: onto the selected instance's best open compatible port; with nothing selected, onto the first (best) open port or loose.
  void add_part(const std::string& part) {
    std::string to;
    if (const design::Instance* s = sel_instance()) to = s->id;
    dreq("edit", design::request("library.drop", "{\"silent\":true,\"part\":" + q(part) + (to.empty() ? "" : ",\"to\":" + q(to)) +
                                                       (to.empty() ? ",\"point\":" + point_json(free_drop_point()) : "") + "}"));
  }
  b3::V3 free_drop_point() const {
    b3::V3 c{orbit_.target[0], orbit_.target[1], orbit_.target[2]};
    return c;
  }
  static std::string point_json(b3::V3 p) {
    char b[120];
    std::snprintf(b, sizeof b, "[%.6f,%.6f,%.6f]", p.x, p.y, p.z);
    return b;
  }

  std::string search_req() {
    lib_.dirty = false;
    if (lib_.category == libui::kMyParts) return design::request("library.my_parts", "{}");
    if (lib_.ranges_for != lib_.category || lib_.ranges.empty()) {
      lib_.ranges_for = lib_.category;
      dreq("ranges", design::request("library.ranges", "{\"category\":" + q(lib_.category) + "}"), true);
    }
    return design::request("library.search", "{\"silent\":true,\"text\":" + q(lib_.query) + ",\"category\":" + q(lib_.category) + ",\"filters\":" + lib_.filters_json() + ",\"limit\":150}");
  }

  static b3::Box box_from(const design::Json& b) {
    double c[3], r[9], h[3];
    design::vec_from(b.at("c"), c, 3), design::vec_from(b.at("R"), r, 9), design::vec_from(b.at("h"), h, 3);
    b3::Box x;
    x.c = {c[0], c[1], c[2]};
    std::copy(r, r + 9, x.R.m);
    x.h = {h[0], h[1], h[2]};
    return x;
  }

  // ---------------------------------------- LIBRARY page -----------------------------------------------------------------------
  void library_page(float H) {
    libui::Actions a;
    const design::Instance* s = sel_instance();
    int targets = -1;
    if (!lib_.selected.empty() && !lib_.mine.count(lib_.selected)) {
      const auto it = targets_.find(lib_.selected);
      if (it != targets_.end()) targets = static_cast<int>(it->second.list.size());
      else want_targets(lib_.selected);
    }
    libui::page(lib_, *thumbs_, {0, kTop}, {pane_l_, H - kTop - kStatus}, s ? s->id : "", targets, a);
    lib_actions(a);
  }
  void want_targets(const std::string& part) {
    Targets& t = targets_[part];
    if (!t.requested) {
      t.requested = true;
      t.rev = dstate_.rev;
      dreq("targets", design::request("library.targets", "{\"part\":" + q(part) + "}"), true);
    }
  }

  // ---------------------------------------- WIRING / SIMULATE / PROGRAM ------------------------------------------------------
  void wiring_page(float W, float H) {
    flat_.set_tooltips(tip_mode_);
    flat_.autowire_ui = [this] { wiring_tab(sel_instance()); };  // the picker folds AUTOWIRE behind a button of its action row
    flat_.autowire_n = static_cast<int>(decisions_.size());
    const ImVec2 pos{0, kTop}, size{W, H - kTop - kStatus};
    begin_pane("##wire2d", pos, size, true);
    ImGui::SetCursorPos({0, 0});
    flat_.page(size, dstate_, sel_.empty() ? "" : sel_.back(), hidden_);  // the picker, or the flattened canvas (wire2d.cpp)
    ImGui::End();
  }

  void simulate_panes(float W, float H) {
    const float ph = H - kTop - kStatus;
    if (pane_l_ > 0) {
      begin_pane("##sim_l", {0, kTop}, {pane_l_, ph});
      sim_controls();
      if (has_design_ && opts_.design_api) {
        ImGui::Spacing();
        theme::caption("DESIGN CHECK RUN");
        simulate_tab();
      }
      ImGui::End();
    }
    if (pane_r_ > 0) {
      begin_pane("##sim_r", pane_pos(true, W), {pane_r_, ph});
      prim_inspector();
      ImGui::End();
    }
    viewport_overlay();
  }

  void sim_controls() {
    theme::caption("SIMULATION");
    ImGui::Text("tick %llu   t = %.4f s", static_cast<unsigned long long>(cur_.tick), static_cast<double>(cur_.tick) * cur_.dt);
    if (cur_.diverged) ImGui::TextColored(theme::kErr, "solver diverged");
    else if (run_.live()) ImGui::TextColored(theme::kAccent, "%s", run_.live_label().c_str());
    else if (cur_.running) ImGui::Text("running (rate %.3g)", cur_.rate);
    else ImGui::TextDisabled("paused");
    if (run_.live()) {  // a program steps this World: Run / Step would fight it
      if (theme::button("STOP PROGRAM", true)) run_.toggle();
      ImGui::SetItemTooltip("kills the program; the World stays where it stopped");
      ImGui::Checkbox("FOLLOW VEHICLE", &run_.follow);
      ImGui::SetItemTooltip("the chase camera keeps the vehicle in view, clear of trees; orbit / pan / zoom the view to take over, tick this to resume");
    } else if (cur_.running) {
      if (theme::button("PAUSE")) send(env_pause(kSource), "pause");
    } else if (theme::button("RUN", true)) {
      send(env_run(kSource, rate_), "run");
    }
    ImGui::BeginDisabled(run_.live());
    ImGui::SameLine();
    ImGui::SetNextItemWidth(70);
    if (ImGui::InputDouble("rate", &rate_, 0, 0, "%.3g", ImGuiInputTextFlags_EnterReturnsTrue) && cur_.running) send(env_run(kSource, rate_), "run");
    ImGui::SetItemTooltip("sim-seconds per wall-second; 0 = as fast as possible (Enter applies while running)");
    if (ImGui::Button("STEP")) send(env_step(kSource, std::max(1, steps_)), "step");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90);
    ImGui::InputInt("ticks", &steps_);
    ImGui::EndDisabled();
    if (!status_.empty()) ImGui::TextColored(theme::kWarn, "%s", status_.c_str());
    run_.draw_result();  // the last full run's verdict + monitors (its final scene is still in the World)
    if (run_.back_to_program) run_.back_to_program = false, set_mode("program");
    ImGui::Spacing();
    theme::caption("SCENE");
    ImGui::BeginChild("##scenetree", {0, has_design_ && opts_.design_api ? 260.f : 0.f});
    for (const auto& [k, child] : tree_.kids) draw_node(k, child);
    ImGui::EndChild();
  }

  void prim_inspector() {
    theme::caption("INSPECTOR");
    if (selected_.empty()) ImGui::TextDisabled("select a prim in the scene tree");
    else ImGui::TextWrapped("%s", selected_.c_str());
    for (const Shown& s : shown_) {
      const std::string unit = s.unit.empty() ? "" : " [" + s.unit + "]";
      if (s.values.size() == 1 && editable(s.attr)) {
        double v = s.values[0];
        ImGui::SetNextItemWidth(140);
        if (ImGui::InputDouble((s.attr + unit).c_str(), &v, 0, 0, "%.6g", ImGuiInputTextFlags_EnterReturnsTrue))
          send(env_setattr(kSource, selected_, s.attr, {v}), "set " + selected_ + "." + s.attr);
      } else {
        std::string line = s.attr + " =";
        char buf[32];
        for (double v : s.values) {
          std::snprintf(buf, sizeof buf, " %.6g", v);
          line += buf;
        }
        ImGui::TextWrapped("%s%s", line.c_str(), unit.c_str());
      }
    }
  }

  void program_page(float W, float H) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});  // the IDE lays out its own panes edge to edge
    begin_pane("##program", {0, kTop}, {W, H - kTop - kStatus}, true);
    ImGui::PopStyleVar();
    prog_.draw(log_, opts_.design_api != nullptr);
    ImGui::End();
  }
  void log_line(const std::string& s) {
    char t[24];
    std::snprintf(t, sizeof t, "%8.2f  ", ImGui::GetTime());
    log_.push_back(t + s);
    if (log_.size() > 500) log_.pop_front();
  }

  // The DRC list, as the body of the DRC tab (compact) and with its detail.
  void drc_body(bool compact) {
    std::size_t errors = 0, warns = 0, infos = 0;
    for (const design::Finding& f : design_.findings) {
      if (f.severity == "error") ++errors;
      else if (f.severity == "warning") ++warns;
      else ++infos;
    }
    ImGui::TextColored(errors ? theme::kErr : theme::kOk, "%zu errors", errors);
    ImGui::SameLine();
    ImGui::TextColored(theme::kWarn, "%zu warnings", warns);
    ImGui::SameLine();
    ImGui::TextDisabled("%zu info", infos);
    if (!new_keys_.empty()) {
      ImGui::SameLine();
      ImGui::TextDisabled("   %zu new", new_keys_.size());
    }
    if (ImGui::BeginTable("drc", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable, {0, compact ? ImGui::GetContentRegionAvail().y - 120 : 0.f})) {
      ImGui::TableSetupColumn("sev", ImGuiTableColumnFlags_WidthFixed, 56);
      ImGui::TableSetupColumn("code / where", ImGuiTableColumnFlags_WidthFixed, 190);
      ImGui::TableSetupColumn("message", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      int n = 0;
      for (const design::Finding& f : design_.findings) {
        const ImVec4 col = f.severity == "error" ? ImVec4{1, 0.35f, 0.35f, 1}
                           : f.severity == "warning" ? ImVec4{1, 0.8f, 0.3f, 1}
                                                     : ImVec4{0.7f, 0.8f, 1, 1};
        ImGui::PushID(n);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        const bool fresh = new_keys_.count(key(f.code, f.where)) > 0;
        if (ImGui::Selectable((f.severity.substr(0, 4) + (fresh ? "*" : "")).c_str(), finding_ == n, ImGuiSelectableFlags_SpanAllColumns)) d_finding(n);
        ImGui::PopStyleColor();
        if (!f.fix.empty()) ImGui::SetItemTooltip("fix: %s", f.fix.c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted((f.code + " " + join(f.instances.empty() ? f.where : f.instances, ",")).c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(f.message.c_str());
        ImGui::PopID();
        ++n;
      }
      ImGui::EndTable();
    }
    if (finding_ >= 0 && finding_ < static_cast<int>(design_.findings.size())) {
      const design::Finding& f = design_.findings[finding_];
      ImGui::Separator();
      ImGui::TextWrapped("%s", f.message.c_str());
      if (!f.fix.empty()) ImGui::TextWrapped("fix: %s", f.fix.c_str());
      for (const std::string& i : f.instances) {
        ImGui::SameLine();
        if (ImGui::SmallButton(i.c_str())) set_sel({i});
      }
    }
  }

  void draw_node(const std::string& name, const Node& n) {
    ImGuiTreeNodeFlags fl = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (n.kids.empty()) fl |= ImGuiTreeNodeFlags_Leaf;
    if (n.prim && n.path == selected_) fl |= ImGuiTreeNodeFlags_Selected;
    const bool open = ImGui::TreeNodeEx(n.path.c_str(), fl, "%s", name.c_str());
    if (ImGui::IsItemClicked() && n.prim && n.path != selected_) {
      select(n.path);
      sel_.clear();  // sel_instance() then falls back to the instance whose body is this prim
      for (const design::Instance& i : design_.instances)
        if (n.path.size() > i.body.size() && n.path.ends_with("/" + i.body)) sel_ = {i.id};
    }
    if (open) {
      for (const auto& [k, child] : n.kids) draw_node(k, child);
      ImGui::TreePop();
    }
  }

  // ================================ per frame ===========================================================================================
  void ui() {
    const ImGuiIO& io = ImGui::GetIO();
    const float W = io.DisplaySize.x, H = io.DisplaySize.y;
    const double now = ImGui::GetTime();
    if (cam_anim_.active) step_cam_anim(now);
    if (opts_.camera) {
      CameraCtl c;
      for (int k = 0; k < 3; ++k) c.target[k] = orbit_.target[k];
      c.yaw = orbit_.yaw, c.pitch = orbit_.pitch, c.dist = orbit_.dist;
      if (opts_.camera(c)) {
        for (int k = 0; k < 3; ++k) orbit_.target[k] = c.target[k];
        orbit_.yaw = c.yaw, orbit_.pitch = c.pitch, orbit_.dist = c.dist;
        cam_anim_.active = false;
      }
    }
    layout(W, H);
    smooth_viewport(io.DeltaTime);
    view_cam_ = view_camera(W, H);
    if (opts_.design_api) {  // cables stay visible at any zoom in the design views; SIMULATE shows true widths and the outdoor sky
      const float want = emph_wires_ && mode_ != Mode::Simulate ? 4.f : 0.f;
      const RenderStyle::Sky sky = mode_ == Mode::Simulate ? RenderStyle::Sky::outdoor : RenderStyle::Sky::studio;
      RenderStyle st = default_render_style();
      if (st.wire_px != want || st.sky != sky) {
        st.wire_px = want;
        st.sky = sky;
        set_default_render_style(st);
      }
    }
    tree_hover_.clear();
    libui::Actions la;
    if (!pending_mode_.empty() && set_mode(pending_mode_)) pending_mode_.clear();
    libui::tick_typing(lib_, now, la);
    if (opts_.design_api && lib_.dirty && !lib_busy_) {
      lib_busy_ = true;
      dreq("search", [this] { return search_req(); }, true);
    }
    if (!opts_.clean) {
      popover_keys();
      top_bar(W);
      ribbon_bar(W);
      switch (mode_) {
        case Mode::Build: build_panes(W, H); break;
        case Mode::Library: library_page(H); break;
        case Mode::Wiring: wiring_page(W, H); break;
        case Mode::Simulate: simulate_panes(W, H); break;
        case Mode::Program: program_page(W, H); break;
      }
      status_bar(W, H);
      confirm_popup();
      close_popup();
      export_popup();
    }
    if (!full_page_) viewport_layer(W, H, now);
    if (!opts_.clean) {  // (last, so the popover is the top-most window; the report then sees every window of the frame)
      controls_popover();
      report_ui();
    }
    if (opts_.overlay && !full_page_) {
      OverlayCtx o;
      o.draw_list = ImGui::GetBackgroundDrawList();
      o.x = vp_.x, o.y = vp_.y, o.w = vp_.z - vp_.x, o.h = vp_.w - vp_.y;
      for (int k = 0; k < 3; ++k) o.cam_pos[k] = view_cam_.pos[k], o.cam_look[k] = view_cam_.lookat[k];
      o.cam_fovy = view_cam_.fovy;
      o.window_w = W, o.window_h = H;
      o.tick = cur_.tick;
      o.sim_time = static_cast<double>(cur_.tick) * cur_.dt;
      opts_.overlay(o);
    }
  }

  void step_cam_anim(double now) {
    const double u = cam_anim_.dur > 1e-3 ? std::clamp((now - cam_anim_.t0) / cam_anim_.dur, 0.0, 1.0) : 1.0;
    const double e = cam_anim_.linear ? u : u * u * (3 - 2 * u);
    if (cam_anim_.fly) {  // eye and look-at along straight lines; the orbit parameters follow from them
      const b3::V3 eye = cam_anim_.eye0 + (cam_anim_.eye1 - cam_anim_.eye0) * e, look = cam_anim_.look0 + (cam_anim_.look1 - cam_anim_.look0) * e;
      const b3::V3 d = eye - look;
      const double r = std::max(1e-6, b3::len(d));
      for (int k = 0; k < 3; ++k) orbit_.target[k] = look[k];
      orbit_.dist = r;
      orbit_.yaw = std::atan2(d.y, d.x);
      orbit_.pitch = std::clamp(std::asin(std::clamp(d.z / r, -1.0, 1.0)), -1.5, 1.5);
      if (u >= 1) cam_anim_.active = false;
      return;
    }
    orbit_.yaw = cam_anim_.sweep ? cam_anim_.yaw0 + (cam_anim_.yaw1 - cam_anim_.yaw0) * e
                                 : cam_anim_.yaw0 + std::remainder(cam_anim_.yaw1 - cam_anim_.yaw0, 6.283185307179586) * e;
    orbit_.pitch = cam_anim_.pitch0 + (cam_anim_.pitch1 - cam_anim_.pitch0) * e;
    orbit_.dist = cam_anim_.dist0 + (cam_anim_.dist1 - cam_anim_.dist0) * e;
    if (cam_anim_.move_target)
      for (int k = 0; k < 3; ++k) orbit_.target[k] = cam_anim_.target_from[k] + (cam_anim_.target_to[k] - cam_anim_.target_from[k]) * e;
    if (u >= 1) cam_anim_.active = false;
  }

  // Everything drawn over / done in the 3D viewport: selection, hover, gizmo, guides, the drag-and-drop ghost, camera navigation.
  void viewport_layer(float W, float H, double now) {
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const b3::Cam cam = b3::Cam::make(view_cam_, W, H);
    const ImVec2 m = io.MousePos;
    const bool in_vp = m.x >= vp_.x && m.x < vp_.z && m.y >= vp_.y && m.y < vp_.w;
    const bool popup = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
    const bool dragging_part = part_drag_active();
    const bool free_mouse = in_vp && !io.WantCaptureMouse && !popup && !dragging_part && !anim_.active;
    dl->PushClipRect({vp_.x, vp_.y}, {vp_.z, vp_.w}, true);
    const bool build = mode_ == Mode::Build && has_design_;
    b3::GizmoOut go;
    if (build) {
      const design::Instance* sel = sel_instance();
      const auto git = sel ? scene_.find(sel->id) : scene_.end();
      const bool can_edit = opts_.design_api && git != scene_.end() && sel->id != dstate_.root && measure_ == 0 && gizmo_.mode != b3::Gizmo::Off;
      // hover + selection marks
      if (free_mouse && !gizmo_.dragging() && !nav_drag_) hover_inst_ = b3::pick(cam, m, scene_, "", &hidden_).id;
      else if (!free_mouse && !gizmo_.dragging()) hover_inst_.clear();
      if (!tree_hover_.empty() && !gizmo_.dragging()) hover_inst_ = tree_hover_;  // the browser row under the pointer
      if (tip_force_until_ > now && scene_.count(tip_force_)) hover_inst_ = tip_force_;  // scripted ui.hover
      if (hover_inst_ != hover_prev_) { hover_prev_ = hover_inst_; hover_since_ = now; }
      // CAD blue: the pre-highlight of the part under the pointer is a thin box outline with a faint wash, the selection a heavier
      // outline of every shape with a brighter wash
      for (const auto& [id, g] : scene_) {
        const bool s = is_selected(id), h = id == hover_inst_ && !s;
        if ((!s && !h) || hidden_.count(id)) continue;
        for (const b3::Box& b : g.boxes) b3::draw_box(dl, cam, b, theme::u32(theme::kAccent, s ? 0.95f : 0.55f), s ? 1.7f : 1.1f);
        b3::draw_brackets(dl, cam, g.lo, g.hi, theme::u32(theme::kAccent, s ? 1.f : 0.7f), s ? 1.8f : 1.2f);
      }
      draw_measure(dl, cam);
      if (can_edit) {
        go = gizmo_.update(cam, scene_, git->second, snap_, free_mouse && !nav_drag_, io.KeyCtrl, vp_);
        if (go.released) finish_gizmo(go, now);
      }
      if (flash_until_ > now && snap_.guides) b3::draw_guides(dl, cam, flash_guides_, static_cast<float>(std::min(1.0, (flash_until_ - now) / 0.5)));
      // open ports of the hovered / selected part: where something could be mated
      if (const auto h = scene_.find(hover_inst_); h != scene_.end() && !gizmo_.dragging())
        for (const b3::Port& p : h->second.ports) {
          ImVec2 s;
          if (!p.used && cam.project(p.pos, s)) dl->AddCircle(s, 4.f, theme::u32(theme::kAccent, 0.7f), 12, 1.2f);
        }
    }
    // navigation + picking
    if (!go.hot || !build) navigation(cam, free_mouse);
    if (m.x != mouse_prev_.x || m.y != mouse_prev_.y) mouse_move_t_ = now;
    mouse_prev_ = m;
    if (build && !hover_inst_.empty() && !go.dragging && !go.hot && now - hover_since_ > 0.35 && !part_drag_active()) {
      const bool forced = tip_force_until_ > now && hover_inst_ == tip_force_;
      if (forced) {  // anchored at the part, offset up and to the right so the card does not hide it
        const auto g = scene_.find(hover_inst_);
        ImVec2 s;
        if (g != scene_.end() && cam.project((g->second.lo + g->second.hi) * 0.5, s)) hover_tip({s.x + 20, s.y - 60});
      } else if (tip_allowed(now) && !nav_drag_ && free_mouse) {
        hover_tip(m);
      }
    }
    drag_drop_layer(cam, dl, now);
    dl->PopClipRect();
    if (build) shortcuts();
  }

  bool part_drag_active() const {
    const ImGuiPayload* pl = ImGui::GetDragDropPayload();
    return (pl && pl->IsDataType("DVSIM_PART") && pl->Data) || anim_.active;
  }

  void navigation(const b3::Cam& cam, bool free_mouse) {
    ImGuiIO& io = ImGui::GetIO();
    const double per_px = 2 * orbit_.dist * std::tan(view_cam_.fovy * 3.14159265358979 / 360) / std::max(1.f, io.DisplaySize.y);
    if (free_mouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      press_ = true;
      press_moved_ = false;
      press_pos_ = io.MousePos;
    }
    if (free_mouse && (ImGui::IsMouseClicked(ImGuiMouseButton_Right) || ImGui::IsMouseClicked(ImGuiMouseButton_Middle))) pan_ = true;
    if (pan_) {
      if (ImGui::IsMouseDown(ImGuiMouseButton_Right) || ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
        for (int k = 0; k < 3; ++k) orbit_.target[k] += (-cam.r[k] * io.MouseDelta.x + cam.u[k] * io.MouseDelta.y) * per_px;
        cam_anim_.active = false;
      } else {
        pan_ = false;
      }
    }
    nav_drag_ = pan_;
    if (press_) {
      if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        if (!press_moved_ && std::hypot(io.MousePos.x - press_pos_.x, io.MousePos.y - press_pos_.y) > 4) press_moved_ = true;
        if (press_moved_ && nav_pan_) {
          for (int k = 0; k < 3; ++k) orbit_.target[k] += (-cam.r[k] * io.MouseDelta.x + cam.u[k] * io.MouseDelta.y) * per_px;
          cam_anim_.active = false;
          nav_drag_ = true;
        } else if (press_moved_) {
          orbit_.yaw -= 0.01 * io.MouseDelta.x;
          orbit_.pitch = std::clamp(orbit_.pitch + 0.01 * io.MouseDelta.y, -1.5, 1.5);
          cam_anim_.active = false;
          nav_drag_ = true;
        }
      } else {
        if (!press_moved_ && mode_ == Mode::Build && has_design_) click_select(cam, press_pos_);
        press_ = false;
      }
    }
    if (free_mouse && io.MouseWheel != 0) {
      orbit_.dist = std::max(0.05, orbit_.dist * std::pow(0.9, io.MouseWheel));
      cam_anim_.active = false;
    }
    if ((nav_drag_ && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) || (free_mouse && io.MouseWheel != 0)) run_.follow = false;  // the user took the camera
  }

  // MEASURE readout: a dimension line between the two parts' box centres with end ticks, the centre distance and the clear gap
  // between the boxes (0 = touching / overlapping) in a small label; while only A is picked, a hint follows it.
  void draw_measure(ImDrawList* dl, const b3::Cam& cam) {
    if (measure_ == 0) return;
    const ImU32 col = theme::u32(theme::kAccent);
    const auto ga = scene_.find(measure_a_), gb = scene_.find(measure_b_);
    ImGui::PushFont(nullptr, theme::kFontSmall);
    if (ga == scene_.end() || gb == scene_.end()) {
      const char* hint = measure_a_.empty() ? "MEASURE: click the first part" : "MEASURE: click the second part";
      const ImVec2 ts = ImGui::CalcTextSize(hint);
      dl->AddRectFilled({vp_.x + 16, vp_.y + 14}, {vp_.x + 28 + ts.x, vp_.y + 22 + ts.y}, theme::u32(theme::kBar, 0.92f), 3);
      dl->AddText({vp_.x + 22, vp_.y + 18}, col, hint);
      ImGui::PopFont();
      return;
    }
    const b3::V3 ca = (ga->second.lo + ga->second.hi) * 0.5, cb = (gb->second.lo + gb->second.hi) * 0.5;
    double gap2 = 0;
    for (int k = 0; k < 3; ++k) {
      const double d = std::max({0.0, gb->second.lo[k] - ga->second.hi[k], ga->second.lo[k] - gb->second.hi[k]});
      gap2 += d * d;
    }
    const double dist = b3::len(cb - ca) * 1000, gap = std::sqrt(gap2) * 1000;
    ImVec2 pa, pb;
    if (cam.project(ca, pa) && cam.project(cb, pb)) {
      dl->AddLine(pa, pb, theme::u32(theme::kBar, 0.9f), 4.f);
      dl->AddLine(pa, pb, col, 1.8f);
      for (const ImVec2 p : {pa, pb}) dl->AddCircleFilled(p, 4.f, col, 12);
      char t[96];
      std::snprintf(t, sizeof t, "%.1f mm   (dx %.1f  dy %.1f  dz %.1f)   gap %.1f mm", dist, (cb.x - ca.x) * 1000, (cb.y - ca.y) * 1000, (cb.z - ca.z) * 1000, gap);
      const ImVec2 ts = ImGui::CalcTextSize(t), mid{(pa.x + pb.x) / 2, (pa.y + pb.y) / 2};
      const ImVec2 p0{std::clamp(mid.x - ts.x / 2 - 6, vp_.x + 8, vp_.z - ts.x - 20), std::clamp(mid.y - ts.y - 12, vp_.y + 8, vp_.w - 30)};
      dl->AddRectFilled(p0, {p0.x + ts.x + 12, p0.y + ts.y + 8}, theme::u32(theme::kBar, 0.95f), 3);
      dl->AddRect(p0, {p0.x + ts.x + 12, p0.y + ts.y + 8}, col, 3);
      dl->AddText({p0.x + 6, p0.y + 4}, theme::u32(theme::kText), t);
    }
    ImGui::PopFont();
  }

  void click_select(const b3::Cam& cam, ImVec2 px) {
    const std::string id = b3::pick(cam, px, scene_, "", &hidden_).id;
    const ImGuiIO& io = ImGui::GetIO();
    if (measure_ > 0) {  // MEASURE: the first click picks A, the second B, the next one starts over
      if (id.empty()) return;
      if (measure_a_.empty() || !measure_b_.empty()) measure_a_ = id, measure_b_.clear();
      else if (id != measure_a_) measure_b_ = id;
      measure_ = measure_b_.empty() ? 2 : 3;  // 1 = pick A, 2 = pick B, 3 = result shown
      return;
    }
    std::vector<std::string> next = io.KeyShift || io.KeyCtrl ? sel_ : std::vector<std::string>{};
    if (!id.empty() && !std::erase(next, id)) next.push_back(id);
    set_sel(next);
  }

  // Compact hover card: name, "id . mass", three key parameters; offset from `anchor` (the cursor, or the part for a scripted hover) so it
  // never sits under the pointer. When it is shown at all: tip_mode_ (ui.view) and tip_allowed().
  void hover_tip(ImVec2 anchor) {
    const auto it = std::find_if(design_.instances.begin(), design_.instances.end(), [&](const design::Instance& i) { return i.id == hover_inst_; });
    if (it == design_.instances.end()) return;
    want_info(it->part);
    const auto inf = lib_.infos.find(it->part);
    std::string name = inf != lib_.infos.end() && !inf->second.name.empty() ? inf->second.name : it->part;
    if (name.size() > 34) name = name.substr(0, 33) + "...";
    ImGui::SetNextWindowPos({anchor.x + 18, anchor.y + 22});
    ImGui::SetNextWindowSizeConstraints({0, 0}, {250, 1e4f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {10, 7});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 4);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8, 3});
    ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4{0.045f, 0.045f, 0.052f, 0.93f});
    ImGui::BeginTooltip();
    ImGui::PushFont(nullptr, 14.f);
    ImGui::TextUnformatted(name.c_str());
    ImGui::PopFont();
    ImGui::PushFont(nullptr, 12.f);
    ImGui::TextDisabled("%s  |  %.1f g", it->id.c_str(), it->mass * 1000);
    if (inf != lib_.infos.end()) {
      int shown = 0;
      for (const libui::ParamRow& p : inf->second.params) {
        if (shown >= 3) break;
        if (p.name == "mass" || p.value == "(table)" || p.value.empty()) continue;
        ImGui::Text("%s", p.name.c_str());
        ImGui::SameLine(112);
        ImGui::TextDisabled("%s %s", p.value.c_str(), p.unit.c_str());
        ++shown;
      }
    }
    ImGui::PopFont();
    ImGui::EndTooltip();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
  }
  // Hover cards appear for a person who is moving the mouse (auto), always (on) or never (off); a scripted ui.hover overrides.
  bool tip_allowed(double now) const { return tip_mode_ == 1 || (tip_mode_ == 0 && now - mouse_move_t_ < 10.0); }

  void shortcuts() {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput || !opts_.design_api || mode_ == Mode::Program) return;  // (the PROGRAM page's editor owns the keyboard)
    if (esc_used_) return;  // (that Esc closed the controls popover, see popover_keys)
    const bool ctrl = io.KeyCtrl;
    if (ctrl) {
      if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) d_undo(io.KeyShift);
      else if (ImGui::IsKeyPressed(ImGuiKey_Y, false)) d_undo(true);
      else if (ImGui::IsKeyPressed(ImGuiKey_D, false) && !sel_.empty()) dreq("edit", design::request("duplicate", "{\"silent\":true,\"instance\":" + q(sel_.back()) + "}"));
      else if (ImGui::IsKeyPressed(ImGuiKey_A, false)) {
        std::vector<std::string> all;
        for (const auto& [id, g] : scene_) all.push_back(id);
        set_sel(all);
      }
      return;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_G, false)) gizmo_.mode = b3::Gizmo::Move, measure_ = 0;
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) gizmo_.mode = b3::Gizmo::Rotate, measure_ = 0;
    if (ImGui::IsKeyPressed(ImGuiKey_F, false)) frame_selection();
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      if (gizmo_.dragging()) gizmo_.cancel();
      else if (measure_ > 0) measure_ = 0, measure_a_.clear(), measure_b_.clear();
      else set_sel({});
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Delete, false) || ImGui::IsKeyPressed(ImGuiKey_X, false)) {
      std::string ops;
      for (const std::string& id : sel_)
        if (id != dstate_.root) ops += (ops.empty() ? "" : ",") + std::string("{\"op\":\"remove\",\"id\":") + q(id) + "}";
      if (!ops.empty()) dreq("edit", edit_req(ops));  // one undo step for the whole selection
    }
    static const std::pair<ImGuiKey, const char*> views[] = {{ImGuiKey_Keypad1, "front"}, {ImGuiKey_Keypad3, "side"}, {ImGuiKey_Keypad7, "top"},
                                                              {ImGuiKey_Keypad5, "iso"}, {ImGuiKey_Keypad0, "iso"},  {ImGuiKey_1, "front"},
                                                              {ImGuiKey_2, "side"},      {ImGuiKey_3, "top"},        {ImGuiKey_4, "iso"}};
    for (const auto& [k, name] : views)
      if (ImGui::IsKeyPressed(k, false)) camera_preset(name);
  }

  // A gizmo drag ended: send it (one request = one undo step), flash the guides that held it.
  void finish_gizmo(const b3::GizmoOut& go, double now) {
    flash_guides_ = go.guides;
    flash_until_ = now + 1.0;
    const std::string ids = json_list(sel_);
    char b[200];
    std::function<std::string(bool)> make;
    if (go.rotate) {
      std::snprintf(b, sizeof b, "\"axis\":[%.6f,%.6f,%.6f],\"angle_deg\":%.6f", go.axis.x, go.axis.y, go.axis.z, go.angle);
      make = [ids, s = std::string(b)](bool un) { return design::request("gizmo.rotate", "{\"silent\":true,\"instances\":" + ids + "," + s + (un ? ",\"unmate\":true" : "") + "}"); };
    } else {
      std::snprintf(b, sizeof b, "\"delta\":[%.6f,%.6f,%.6f]", go.delta.x, go.delta.y, go.delta.z);
      make = [ids, s = std::string(b)](bool un) { return design::request("gizmo.move", "{\"silent\":true,\"instances\":" + ids + "," + s + (un ? ",\"unmate\":true" : "") + "}"); };
    }
    bool held = false;
    for (const std::string& i : sel_) held = held || is_held(i);
    if (held) confirm_unmate(sel_.back(), make);
    else dreq("edit", make(false));
  }

  // ---- drag and drop of a library part into the viewport (user drag from a card, or a scripted `library.drag` animation) ----
  struct Target {
    std::string to, port;
    b3::V3 pos, port_pos;
    b3::Mat3 R;
  };
  struct Targets {
    bool requested = false, loaded = false;
    std::string rev;
    std::vector<b3::Box> boxes;
    std::vector<Target> list;
  };
  struct Ghost {
    bool valid = false, snapped = false;
    std::string to, port;
    b3::V3 pos;
    b3::Mat3 R;
  };
  struct Anim {  // a scripted drag: the cursor travels from `from` to `to_px`, then (then == "drop") the part is dropped
    bool active = false, ready = false;
    std::string part, to, port, then;
    bool has_point = false;
    b3::V3 point;
    ImVec2 from{0, 0}, to_px{0, 0}, cursor{0, 0};
    double t0 = 0, dur = 1.5, requested_at = 0;
  };

  Ghost ghost_for(const b3::Cam& cam, const std::string& part, ImVec2 mouse, double progress) {
    Ghost g;
    const auto it = targets_.find(part);
    if (it == targets_.end() || !it->second.loaded) return g;
    const Targets& t = it->second;
    g.valid = true;
    double best = 70;
    const Target* pick = nullptr;
    for (const Target& x : t.list) {
      if (anim_.active && !anim_.to.empty() && progress > 0.8) {
        if (x.to == anim_.to && (anim_.port.empty() || x.port == anim_.port)) pick = &x;
        continue;
      }
      if (!snap_.ports) continue;  // the "ports" snapping checkbox / ui.snap: a free drop
      ImVec2 s;
      if (!cam.project(x.port_pos, s)) continue;
      const double d = std::hypot(s.x - mouse.x, s.y - mouse.y);
      if (d < best) { best = d; pick = &x; }
    }
    if (anim_.active && !anim_.to.empty() && progress > 0.8 && !pick)  // that port is not in the list (taken?): any target on that instance
      for (const Target& x : t.list)
        if (x.to.substr(0, x.to.find('.')) == anim_.to.substr(0, anim_.to.find('.'))) { pick = &x; break; }
    if (pick) {
      g.snapped = true;
      g.to = pick->to, g.port = pick->port, g.pos = pick->pos, g.R = pick->R;
      return g;
    }
    // free: the cursor ray onto the scene / the ground; the part rests on what it touches
    const b3::Ray r = cam.ray(mouse);
    const b3::Hit h = b3::pick(cam, mouse, scene_);
    b3::V3 p, n{0, 0, 1};
    bool got = false;
    if (!h.id.empty()) { p = h.point, n = h.normal, got = true; }
    b3::V3 gp;
    if (b3::ray_plane(r, {0, 0, 0}, {0, 0, 1}, gp) && (!got || b3::len(gp - r.o) < h.t)) { p = gp, n = {0, 0, 1}, got = true; }
    if (!got) { g.valid = false; return g; }
    double lowest = 0, ext = 0.01;
    for (const b3::Box& b : t.boxes) {
      lowest = std::min(lowest, b.c.z - (std::abs(b.R.m[6]) * b.h.x + std::abs(b.R.m[7]) * b.h.y + std::abs(b.R.m[8]) * b.h.z));
      ext = std::max({ext, b.h.x, b.h.y, b.h.z});
    }
    g.pos = std::abs(n.z) > 0.5 ? b3::V3{p.x, p.y, p.z - lowest * (n.z > 0 ? 1 : -1)} : p + n * ext;
    return g;
  }

  void drag_drop_layer(const b3::Cam& cam, ImDrawList* dl, double now) {
    const ImGuiIO& io = ImGui::GetIO();
    const ImGuiPayload* pl = ImGui::GetDragDropPayload();
    if (pl && pl->IsDataType("DVSIM_MYPART") && pl->Data) return my_part_drag(cam, dl, static_cast<const char*>(pl->Data));
    if (was_my_drag_) {  // (released: my_part_drag has dropped it)
      was_my_drag_ = false;
      ghost_ = {};
    }
    std::string part;
    bool scripted = anim_.active;
    bool user = pl && pl->IsDataType("DVSIM_PART") && pl->Data;
    if (user) part = static_cast<const char*>(pl->Data);
    double progress = 0;
    ImVec2 mouse = io.MousePos;
    if (scripted) {
      part = anim_.part;
      want_targets(part);
      const Targets& t = targets_[part];
      if (!anim_.ready) {
        if (t.loaded || now - anim_.requested_at > 5.0) {  // (no design editor / a failed request: fly anyway)
          anim_.ready = true;
          anim_.t0 = now;
          anim_.to_px = anim_target_px(cam, t);
        }
        mouse = anim_.from;
      } else {
        progress = std::clamp((now - anim_.t0) / anim_.dur, 0.0, 1.0);
        const double e = progress * progress * (3 - 2 * progress);
        mouse = {static_cast<float>(anim_.from.x + (anim_.to_px.x - anim_.from.x) * e), static_cast<float>(anim_.from.y + (anim_.to_px.y - anim_.from.y) * e - 60 * std::sin(3.14159265 * e))};
      }
      anim_.cursor = mouse;
    }
    const bool dragging = user || scripted;
    if (dragging) {
      was_dragging_ = true;
      drag_part_ = part;
      if (user) want_targets(part);
      ghost_in_vp_ = mouse.x >= vp_.x && mouse.x < vp_.z && mouse.y >= vp_.y && mouse.y < vp_.w;
      ghost_ = ghost_in_vp_ && (!scripted || anim_.ready) ? ghost_for(cam, part, mouse, progress) : Ghost{};
      const Targets& t = targets_[part];
      if (ghost_.valid && ghost_in_vp_) {
        for (const Target& x : t.list) {  // compatible seats
          ImVec2 s;
          if (cam.project(x.port_pos, s)) dl->AddCircle(s, 5.5f, theme::u32(theme::kAccent, 0.55f), 14, 1.5f);
        }
        std::vector<b3::Box> boxes = t.boxes;
        if (boxes.empty()) boxes.push_back({{0, 0, 0}, {}, {0.01, 0.01, 0.01}});
        b3::draw_ghost(dl, cam, boxes, ghost_.pos, ghost_.R, ghost_.snapped ? theme::u32(theme::kAccent) : theme::u32({1, 1, 1, 1}, 0.8f));
        if (ghost_.snapped) {
          for (const Target& x : t.list)
            if (x.to == ghost_.to && x.port == ghost_.port) {
              ImVec2 s;
              if (cam.project(x.port_pos, s)) {
                dl->AddCircle(s, 11.f, theme::u32(theme::kAccent), 24, 2.f);
                dl->AddText({s.x + 14, s.y - 8}, theme::u32(theme::kAccent), x.to.c_str());
              }
              break;
            }
        }
      }
      // the chip that follows the cursor (above the panels)
      ImDrawList* fg = ImGui::GetForegroundDrawList();
      const ImVec2 sz = ImGui::CalcTextSize(part.c_str());
      fg->AddRectFilled({mouse.x + 12, mouse.y + 10}, {mouse.x + 28 + sz.x, mouse.y + 14 + sz.y + 6}, theme::u32({0, 0, 0, 0.82f}), 3);
      fg->AddRect({mouse.x + 12, mouse.y + 10}, {mouse.x + 28 + sz.x, mouse.y + 14 + sz.y + 6}, theme::u32(theme::kAccent, 0.8f), 3);
      fg->AddText({mouse.x + 20, mouse.y + 14}, theme::u32(theme::kText), part.c_str());
      if (scripted) {  // a visible cursor for recordings
        const ImVec2 c = mouse;
        fg->AddTriangleFilled(c, {c.x + 11, c.y + 14}, {c.x + 3, c.y + 11}, IM_COL32(255, 255, 255, 240));
        fg->AddTriangle(c, {c.x + 11, c.y + 14}, {c.x + 3, c.y + 11}, IM_COL32(0, 0, 0, 255), 1.f);
      }
    }
    const bool finish_user = was_dragging_ && !scripted && ImGui::IsMouseReleased(ImGuiMouseButton_Left);
    const bool finish_script = scripted && anim_.ready && progress >= 1.0;
    if (finish_user || finish_script) {
      if (ghost_.valid && ghost_in_vp_ && (!finish_script || anim_.then == "drop")) drop_ghost();
      was_dragging_ = false;
      ghost_ = {};
      anim_.active = false;
    }
  }
  // A MY PARTS card dragged over the 3D view: a chip follows the cursor and the drop point (the scene or the ground under it) is
  // marked; released inside the view it is one sub.insert, welded there. (Attaching it by a mate to an exported port is the
  // backend's sub.insert `mate`; the window welds at the point.)
  void my_part_drag(const b3::Cam& cam, ImDrawList* dl, const std::string& path) {
    const ImGuiIO& io = ImGui::GetIO();
    Targets& t = targets_["@my-part"];  // (no ports to snap to: ghost_for gives the free point)
    t.requested = t.loaded = true;
    const bool in_vp = io.MousePos.x >= vp_.x && io.MousePos.x < vp_.z && io.MousePos.y >= vp_.y && io.MousePos.y < vp_.w;
    ghost_ = in_vp ? ghost_for(cam, "@my-part", io.MousePos, 0) : Ghost{};
    was_my_drag_ = true;
    if (ghost_.valid) {
      ImVec2 s;
      if (cam.project(ghost_.pos, s)) {
        dl->AddCircle(s, 9.f, theme::u32(theme::kAccent), 20, 2.f);
        dl->AddLine({s.x - 14, s.y}, {s.x + 14, s.y}, theme::u32(theme::kAccent, 0.7f));
        dl->AddLine({s.x, s.y - 14}, {s.x, s.y + 14}, theme::u32(theme::kAccent, 0.7f));
      }
    }
    const auto mp = lib_.mine.find(path);
    const std::string name = mp != lib_.mine.end() ? mp->second.name : path;
    ImDrawList* fg = ImGui::GetForegroundDrawList();
    const ImVec2 sz = ImGui::CalcTextSize(name.c_str()), m = io.MousePos;
    fg->AddRectFilled({m.x + 12, m.y + 10}, {m.x + 28 + sz.x, m.y + 14 + sz.y + 6}, theme::u32({0, 0, 0, 0.82f}), 3);
    fg->AddRect({m.x + 12, m.y + 10}, {m.x + 28 + sz.x, m.y + 14 + sz.y + 6}, theme::u32(theme::kAccent, 0.8f), 3);
    fg->AddText({m.x + 20, m.y + 14}, theme::u32(theme::kText), name.c_str());
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && ghost_.valid) insert_my_part(path, ghost_.pos);
  }

  ImVec2 anim_target_px(const b3::Cam& cam, const Targets& t) const {
    ImVec2 s{(vp_.x + vp_.z) / 2, (vp_.y + vp_.w) / 2};
    if (!anim_.to.empty()) {
      for (const Target& x : t.list)
        if (x.to == anim_.to && (anim_.port.empty() || x.port == anim_.port)) {
          cam.project(x.port_pos, s);
          return s;
        }
      const auto g = scene_.find(anim_.to.substr(0, anim_.to.find('.')));
      if (g != scene_.end()) cam.project(g->second.pos, s);
    } else if (anim_.has_point) {
      cam.project(anim_.point, s);
    }
    return s;
  }

  void drop_ghost() {
    const std::string args = ghost_.snapped
        ? "{\"silent\":true,\"part\":" + q(drag_part_) + ",\"to\":" + q(ghost_.to) + ",\"port\":" + q(ghost_.port) + "}"
        : "{\"silent\":true,\"part\":" + q(drag_part_) + ",\"point\":" + point_json(ghost_.pos) + "}";
    dreq("edit", design::request("library.drop", args));
  }

  // ---- scripted UI events (backend ui_poll) and script actions ----
  void ui_event(const design::Json& e) {
    const std::string cmd = e.s("cmd");
    const double now = ImGui::GetTime();
    log_line("ui " + cmd);
    if (cmd == "mode") set_mode(e.s("name"));
    else if (cmd == "keys") keys_open_ = e.at("open").kind != design::Json::Bool || e.at("open").num != 0;
    else if (cmd == "program") {  // a source file in the PROGRAM page (text empty = back to the design TOML)
      prog_.show_external(e.s("name"), e.s("text"), e.at("scroll").kind == design::Json::Num ? static_cast<float>(e.at("scroll").num) : -1.f);
      if (mode_ != Mode::Program) set_mode("program");
    } else if (cmd == "prog_cmd" && run_.command(e.s("name"))) {  // scripted Run slot: bench | full | run | stop | target:<id or scenario>
      if (const std::string n = e.s("name"); n != "stop" && n != "follow" && n != "unfollow" && n.rfind("hover:", 0) != 0 && n.rfind("click:", 0) != 0 && mode_ != Mode::Program) set_mode("program");
    } else if (cmd.rfind("prog_", 0) == 0) {  // scripted IDE actions (prog.ui)
      if (mode_ != Mode::Program) set_mode("program");
      prog_.script(e);
    } else if (cmd == "camera_move") {
      camera_move(e);
    } else if (cmd == "visibility") {
      const bool vis = e.at("visible").kind != design::Json::Bool || e.at("visible").num != 0;
      for (const design::Json& i : e.at("instances").arr) {
        if (vis) hidden_.erase(i.str);
        else hidden_.insert(i.str);
      }
    } else if (cmd == "measure") {
      measure_a_ = e.s("a"), measure_b_ = e.s("b");
      measure_ = measure_a_.empty() ? 0 : measure_b_.empty() ? 2 : 3;
      if (measure_ > 0 && mode_ != Mode::Build) set_mode("build");
    } else if (cmd == "wire_pick") {
      std::vector<std::string> ids;
      for (const design::Json& i : e.at("instances").arr) ids.push_back(i.str);
      flat_.pick(ids, dstate_);
      if (mode_ != Mode::Wiring) set_mode("wiring");
    } else if (cmd == "wire_view") {
      if (!e.s("panel").empty()) flat_.set_side_open(e.s("panel") == "open");
      flat_.view(e.n("zoom", 0), e.at("fit").num != 0, e.n("cx", wire2d::Canvas::kKeep), e.n("cy", wire2d::Canvas::kKeep), e.n("seconds", 0));
    } else if (cmd == "camera") {
      if (!e.s("preset").empty()) camera_preset(e.s("preset"));
      else cam_anim_ = {true, now, orbit_.yaw, orbit_.pitch, orbit_.dist, e.n("yaw", orbit_.yaw), e.n("pitch", orbit_.pitch), e.n("dist", orbit_.dist)};
    } else if (cmd == "frame") {
      std::vector<std::string> ids;
      for (const design::Json& i : e.at("instances").arr) ids.push_back(i.str);
      if (!ids.empty()) set_sel(ids);
      frame_selection();
    } else if (cmd == "gizmo") {
      const std::string m = e.s("mode");
      gizmo_.mode = m == "rotate" ? b3::Gizmo::Rotate : m == "none" ? b3::Gizmo::Off : b3::Gizmo::Move;
    } else if (cmd == "view") {
      const std::string t = e.s("tooltips");
      tip_mode_ = t == "on" ? 1 : t == "off" ? 2 : 0;
      emph_wires_ = e.at("emphasise_wires").kind == design::Json::Bool || e.at("emphasise_wires").kind == design::Json::Num ? e.at("emphasise_wires").num != 0 : emph_wires_;
    } else if (cmd == "toast") {
      flat_.notify(e.s("text"), e.at("bad").num != 0);
    } else if (cmd == "hover") {
      tip_force_ = e.s("instance");
      tip_force_until_ = now + e.n("seconds", 6.0);
      hover_since_ = now - 1;
      if (mode_ != Mode::Build) set_mode("build");
    } else if (cmd == "guides") snap_.guides = e.at("on").num != 0;
    else if (cmd == "snap") {
      snap_.grid = e.n("grid_m", snap_.grid), snap_.angle = e.n("angle_deg", snap_.angle);
      snap_.edges = e.at("snap_edges").num != 0, snap_.ports = e.at("snap_ports").num != 0;
    } else if (cmd == "select") {
      std::vector<std::string> ids = e.at("add").num != 0 ? sel_ : std::vector<std::string>{};
      for (const design::Json& i : e.at("instances").arr) ids.push_back(i.str);
      sel_synced_ = ids;  // (the backend already knows)
      set_sel(ids);
    } else if (cmd == "search") {
      left_tab_ = 1;  // typing happens in the PARTS list's search box
      lib_.category = e.s("category");
      lib_.filters.clear();
      for (const auto& [k, r] : e.at("filters").obj)
        if (r.arr.size() >= 2) {
          const auto rg = lib_.ranges.find(k);
          lib_.filters[k] = {true, r.arr[0].kind == design::Json::Num ? static_cast<float>(r.arr[0].num) : rg != lib_.ranges.end() ? static_cast<float>(rg->second.lo) : 0.f,
                             r.arr[1].kind == design::Json::Num ? static_cast<float>(r.arr[1].num) : rg != lib_.ranges.end() ? static_cast<float>(rg->second.hi) : 1e9f};
        }
      const double ms = e.n("type_ms");
      if (ms > 0) {
        lib_.query[0] = 0;
        lib_.type_target = e.s("text");
        lib_.type_ms = ms;
        lib_.type_next = now;
      } else {
        std::snprintf(lib_.query, sizeof lib_.query, "%s", e.s("text").c_str());
      }
      lib_.dirty = true;
    } else if (cmd == "color") {
      const std::string id = e.s("instance");
      set_sel({id});
      if (e.at("rgba").arr.size() >= 3) {
        std::array<float, 4> c{0, 0, 0, 1};
        for (std::size_t i = 0; i < 4 && i < e.at("rgba").arr.size(); ++i) c[i] = static_cast<float>(e.at("rgba").arr[i].num);
        color_pick_[id] = c;
      }
      right_tab_ = 0;
      color_flash_ = now + 1.8;
    } else if (cmd == "move" || cmd == "rotate") {
      set_sel({e.s("instance")});
      flash_guides_.clear();
      for (const design::Json& g : e.at("guides").arr) {
        b3::Guide x;
        x.axis = static_cast<int>(g.n("axis"));
        double v[3];
        design::vec_from(g.at("a"), v, 3), x.a = {v[0], v[1], v[2]};
        design::vec_from(g.at("b"), v, 3), x.b = {v[0], v[1], v[2]};
        flash_guides_.push_back(x);
      }
      flash_until_ = now + 1.4;
    } else if (cmd == "drag") {
      left_tab_ = 1;  // the ghost flies out of the PARTS list
      anim_ = {};
      anim_.active = true;
      anim_.part = e.s("part");
      anim_.to = e.s("to");
      anim_.port = e.s("port");
      anim_.then = e.s("then");
      anim_.dur = e.n("duration", 1.5);
      if (e.at("point").kind == design::Json::Arr) {
        double v[3];
        design::vec_from(e.at("point"), v, 3);
        anim_.has_point = true;
        anim_.point = {v[0], v[1], v[2]};
      }
      if (e.at("from").kind == design::Json::Arr && e.at("from").arr.size() >= 2) anim_.from = {static_cast<float>(e.at("from").arr[0].num), static_cast<float>(e.at("from").arr[1].num)};
      else if (const auto c = lib_.card_at.find(anim_.part); c != lib_.card_at.end()) anim_.from = c->second;
      else anim_.from = {vp_.x > 60 ? 130.f : 60.f, vp_.y + 140};
      if (mode_ == Mode::Library) set_mode("build");  // the drop target is the 3D view
      want_targets(anim_.part);
      anim_.requested_at = now;
    }
  }

  void script_ui(const ScriptCmd& c) {
    if (c.action == "mode") {
      if (!set_mode(c.args[0]) && opts_.design_api) pending_mode_ = c.args[0];  // the first state has not arrived yet: retry each frame
    } else if (c.action == "uisel") {
      std::vector<std::string> ids;
      std::stringstream ss(c.args[0]);
      for (std::string s; std::getline(ss, s, ',');) ids.push_back(s);
      set_sel(ids);
    } else if (c.action == "search") {
      std::string t;
      for (const std::string& a : c.args) t += (t.empty() ? "" : " ") + a;
      std::snprintf(lib_.query, sizeof lib_.query, "%s", t.c_str());
      lib_.dirty = true;
    } else if (c.action == "uicam") camera_preset(c.args[0]);
    else if (c.action == "uiframe") frame_selection();
    else if (c.action == "uikeys") keys_open_ = c.args.empty() ? !keys_open_ : c.args[0] != "off";
    else if (c.action == "crumb") close_docs(static_cast<int>(to_double(c.args[0], "crumb depth")));  // a click on that crumb (0 = base design)
  }

  void rebuild_scene() {
    scene_.clear();
    for (const auto& [id, g] : dstate_.geo) {
      b3::InstGeo o;
      o.id = id;
      o.pos = {g.pos[0], g.pos[1], g.pos[2]};
      std::copy(g.R, g.R + 9, o.R.m);
      o.lo = {g.lo[0], g.lo[1], g.lo[2]};
      o.hi = {g.hi[0], g.hi[1], g.hi[2]};
      for (const design::GeoBox& b : g.boxes) {
        b3::Box x;
        x.c = {b.c[0], b.c[1], b.c[2]};
        std::copy(b.R, b.R + 9, x.R.m);
        x.h = {b.h[0], b.h[1], b.h[2]};
        o.boxes.push_back(x);
      }
      o.has_color = g.has_color;
      std::copy(g.color, g.color + 4, o.color);
      o.root = id == dstate_.root;
      o.free = std::find(dstate_.free.begin(), dstate_.free.end(), id) != dstate_.free.end();
      if (const auto p = dstate_.ports.find(id); p != dstate_.ports.end())
        for (const design::Port& q : p->second)
          if (q.placed && q.kind == "mech") o.ports.push_back({q.name, {q.pos[0], q.pos[1], q.pos[2]}, {q.axis[0], q.axis[1], q.axis[2]}, q.used});
      for (const design::Instance& i : dstate_.report.instances)
        if (i.id == id) o.part = i.part;
      scene_[id] = std::move(o);
    }
  }

  struct CamAnim {
    bool active = false;
    double t0 = 0, yaw0 = 0, pitch0 = 0, dist0 = 0, yaw1 = 0, pitch1 = 0, dist1 = 0;
    b3::V3 target_from, target_to;
    bool move_target = false;
    // camera.orbit / camera.fly (ui_event "camera_move"): duration, easing, a yaw SWEEP (no shortest-way wrap), or a straight fly
    // of the eye and the look-at point
    double dur = 0.5;
    bool linear = false, sweep = false, fly = false;
    b3::V3 eye0, eye1, look0, look1;
  };

  Host& host_;
  ViewerOptions opts_;
  std::vector<ScriptCmd> script_;
  Frame cur_;
  std::uint64_t frame_scene_ = 0;
  Orbit orbit_;
  Node tree_;
  design::Report design_;  // the report of the `dvsim design build` this model came from (with a backend: of the design as edited)
  bool has_design_ = false;
  // Design editing (all driven through opts_.design_api, see viewer.h)
  design::State dstate_;
  std::deque<DReq> dq_;
  wire2d::Canvas flat_{[this](const std::string& tool, const std::string& args, bool quiet) {
    dq_.push_back({tool, quiet, [tool, args] { return design::request(tool, args); }});
  }};
  DReq dcur_;
  std::future<std::string> dfut_;
  bool draining_ = false, keep_camera_ = false;
  std::string loaded_rev_, dstatus_, edit_summary_, saved_, sugg_target_, sim_title_, sel_port_, sel_port_inst_;
  std::chrono::steady_clock::time_point last_poll_ = std::chrono::steady_clock::now(), last_ev_ = std::chrono::steady_clock::now();
  Placement place_;
  int cycling_ = -1, finding_ = -1;
  // ---- the split UI (modes, panes, library, 3D interaction) ----
  Mode mode_ = Mode::Simulate;
  ImVec4 vp_{0, 0, 0, 0};  // viewport rectangle x0 y0 x1 y1 (window pixels), eased toward vp_goal_
  ImVec4 vp_goal_{0, 0, 0, 0};
  bool vp_init_ = false;
  float pane_l_ = 0, pane_r_ = 0;
  bool full_page_ = false;
  Camera view_cam_;
  int right_tab_ = 0;  // BUILD right pane: 0 inspect, 1 parts, 2 drc, 3 place
  libui::Model lib_;
  libui::Thumbs* thumbs_ = nullptr;
  bool lib_busy_ = false, snap_requested_ = false;
  std::string pending_mode_;
  std::map<std::string, Targets> targets_;
  b3::Scene scene_;
  b3::Gizmo gizmo_;
  b3::Settings snap_;
  std::vector<std::string> sel_, sel_synced_;
  std::string hover_inst_, hover_prev_;
  double hover_since_ = 0;
  int tip_mode_ = 0;  // ui.view tooltips: 0 auto (while a person moves the mouse), 1 on, 2 off
  std::string tip_force_;  // ui.hover: this instance's card is shown until tip_force_until_
  double tip_force_until_ = 0, mouse_move_t_ = -1e9;
  ImVec2 mouse_prev_{-1, -1};
  bool emph_wires_ = true;  // ui.view emphasise_wires: cables >= 4 px wide in BUILD / LIBRARY / WIRING (SIMULATE keeps true widths)
  int thumb_rev_ = 0;       // backend: library thumbnails finished so far (changes -> ask the library again)
  bool press_ = false, press_moved_ = false, pan_ = false, nav_drag_ = false;
  ImVec2 press_pos_{0, 0};
  std::vector<b3::Guide> flash_guides_;
  double flash_until_ = 0, color_flash_ = 0;
  std::map<std::string, std::array<float, 4>> color_pick_;
  bool color_dirty_ = false;
  CamAnim cam_anim_;
  int left_tab_ = 0;                // BUILD left dock: 0 BROWSER, 1 PARTS (the library list)
  bool framed_ = false;             // the design has been framed once on open
  int measure_ = 0;                 // MEASURE tool: 0 off, 1 pick A, 2 pick B, 3 result
  std::string measure_a_, measure_b_, tree_hover_;
  bool nav_pan_ = false;            // left-drag pans instead of orbiting (view cube panel)
  std::set<std::string> hidden_;    // instances switched off with the eye in the browser
  bool hidden_dirty_ = false;
  bool keys_open_ = false;          // the controls popover under the keys icon of the top bar
  bool esc_used_ = false;           // this frame's Esc closed it (the shortcuts leave it alone)
  struct Rep {                      // rectangles (x0 y0 x1 y1, window px) of the top bar pieces as drawn this frame; see report_ui
    ImVec4 icon{0, 0, 0, 0}, tabs{0, 0, 0, 0}, zoom{0, 0, 0, 0}, pop{0, 0, 0, 0};
    struct CrumbRect {
      std::string label;
      ImVec4 r;
    };
    std::vector<CrumbRect> crumbs;
    std::string cost, from;         // the status bar's cost segment; the inspector's "from <file>" line of the selected sub-assembly item
    std::vector<std::string> panel;  // the per-sub-assembly mass / cost rows of the inspector
  } rep_;
  // sub-assembly workspace (documents, exports, MY PARTS)
  struct CloseAsk {                 // returning up from a dirty document: save / discard / cancel
    bool open = false, want_popup = false;
    int to = 0;
    char path[200] = "";
  } close_ask_;
  struct ExportAsk {                // "Export as...": kind (port | connector | param), ref ("inst.name"), the export's name
    bool want_popup = false;
    int kind = 0;
    char ref[96] = "", name[48] = "";
  } export_ask_;
  int part_seq_ = 0;
  bool close_failed_ = false, was_my_drag_ = false;
  std::size_t last_docs_ = 1;       // the document depth last seen (a change clears the selection and reframes)
  std::string sel_sub_;            // the sub-assembly node clicked in the browser tree (Mirror / Open part act on it)
  std::string rep_sent_, status_msg_;  // the last ui.report sent; the message the status bar showed
  std::size_t chip_errors_ = 0, chip_warns_ = 0;  // the DRC chip's counts (status bar)
  std::size_t edit_errors_ = 0, edit_warns_ = 0;  // the DRC counts after the last edit (what edit_summary_ describes)
  std::string geom_owner_key_;
  std::vector<std::string> geom_owner_;
  std::vector<GeomPose> hidden_poses_;
  std::vector<TubeVisual> shown_tubes_;
  Anim anim_;
  Ghost ghost_;
  bool ghost_in_vp_ = false, was_dragging_ = false;
  std::string drag_part_;
  std::function<std::string(bool)> confirm_;
  std::string confirm_id_;
  bool confirm_open_ = false;
  std::deque<std::string> log_;
  progui::ProgramUi prog_;  // the PROGRAM page (program_ui.cpp)
  runui::RunUi run_{prog_};  // its Test / Run controls (run_ui.cpp)
  std::vector<Sugg> suggs_;
  bool sugg_swap_ = false, auto_dev_ = true;
  std::vector<Decision> decisions_;
  std::vector<std::string> edit_lines_;
  std::vector<std::pair<std::string, std::string>> sim_rows_;
  std::set<std::string> new_keys_, hl_;  // findings new in the last edit / instances the clicked finding involves
  char cat_buf_[64] = "", part_buf_[96] = "", save_buf_[256] = "";
  std::vector<std::string> prim_paths_;
  std::string pending_body_;
  std::string selected_, status_;
  std::vector<Shown> shown_;
  std::vector<std::pair<std::string, std::future<Reply>>> inspect_, cmds_;
  std::future<Reply> prims_;
  double rate_ = 1.0;
  int steps_ = 1;
};
}  // namespace

void run_viewer(Host& host, const ViewerOptions& opts) { Viewer(host, opts).run(); }

std::uint64_t viewer_latest_frame(std::vector<std::uint8_t>* rgba, int* width, int* height) {
  std::lock_guard<std::mutex> lk(g_frame_mu);
  if (rgba) *rgba = g_frame;
  if (width) *width = g_frame_w;
  if (height) *height = g_frame_h;
  return g_frame_seq;
}

std::vector<ScriptCmd> parse_script(const std::string& script) {
  std::vector<ScriptCmd> out;
  std::istringstream entries(script);
  for (std::string entry; std::getline(entries, entry, ';');) {
    std::istringstream words(entry);
    ScriptCmd c;
    std::string frame;
    if (!(words >> frame)) continue;  // empty entry (trailing ';')
    if (!(words >> c.action)) throw std::runtime_error("script entry '" + entry + "': expected FRAME ACTION [ARGS]");
    c.frame = static_cast<int>(to_double(frame, "frame"));
    for (std::string a; words >> a;) c.args.push_back(a);
    if (c.action == "select" || c.action == "inst") {  // UI actions with no bus command: select a prim / a design instance's body
      if (c.args.size() != 1) throw std::runtime_error("script: " + c.action + (c.action == "inst" ? " INSTANCE" : " PATH"));
    } else if (const auto [ulo, uhi] = ui_action_args(c.action); ulo >= 0) {
      const int n = static_cast<int>(c.args.size());
      if (n < ulo || n > uhi) throw std::runtime_error("script entry '" + entry + "': " + c.action + " takes " + std::to_string(ulo) + (uhi > ulo ? ".." + std::to_string(uhi) : "") + " argument(s)");
      if (c.action == "mode" && c.args[0] != "build" && c.args[0] != "library" && c.args[0] != "wiring" && c.args[0] != "simulate" && c.args[0] != "program")
        throw std::runtime_error("script: mode build|library|wiring|simulate|program");
    } else if (const auto [lo, hi] = design_action_args(c.action); lo >= 0) {
      const int n = static_cast<int>(c.args.size());
      if (n < lo || n > hi) throw std::runtime_error("script entry '" + entry + "': " + c.action + " takes " + std::to_string(lo) + (hi > lo ? ".." + std::to_string(hi) : "") + " argument(s)");
    } else {
      script_envelope(c);  // validate now: bad scripts fail before a window opens
    }
    out.push_back(std::move(c));
  }
  return out;
}

std::vector<std::uint8_t> script_envelope(const ScriptCmd& c, const std::string& source) {
  const auto need = [&](std::size_t n, const char* usage) {
    if (c.args.size() != n) throw std::runtime_error(std::string("script: ") + usage);
  };
  if (c.action == "run") {
    need(1, "run RATE");
    return env_run(source, to_double(c.args[0], "rate"));
  }
  if (c.action == "pause") {
    need(0, "pause takes no arguments");
    return env_pause(source);
  }
  if (c.action == "step") {
    need(1, "step N");
    return env_step(source, static_cast<int>(to_double(c.args[0], "N")));
  }
  if (c.action == "setattr") {
    if (c.args.size() < 3) throw std::runtime_error("script: setattr PATH ATTR V [V...]");
    std::vector<double> v;
    for (std::size_t i = 2; i < c.args.size(); ++i) v.push_back(to_double(c.args[i], "value"));
    return env_setattr(source, c.args[0], c.args[1], v);
  }
  throw std::runtime_error("script: unknown action '" + c.action + "' (run, pause, step, setattr, select, inst, d* design actions)");
}
