// M9 exit checks (ctest label `gpu`, names render.*): the headless Vulkan renderer and lidar.
// (The golden-image PSNR tests were dropped 2026-10-11: the golden went stale with every look change. Images are
// checked against llvmpipe in the same process instead, see render_gpu.*.)
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "core/world.h"
#include "flatbuffers/flatbuffers.h"
#include "mujoco/mujoco_solver.h"
#include "dvsim_generated.h"
#include "render/renderer.h"

namespace {
const std::string kScene = DVSIM_TEST_DATA_DIR "/render_scene.xml";
constexpr int kW = 320, kH = 240;
const Camera kCam = {{2.5, -2.2, 1.6}, {0, 0, 0.25}, 45};

std::unique_ptr<Solver> scene() { return load_mujoco(kScene); }

double psnr_rgb(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  EXPECT_EQ(a.size(), b.size());
  double se = 0;
  std::size_t n = 0;
  for (std::size_t i = 0; i + 3 < a.size() && i + 3 < b.size(); i += 4)
    for (int c = 0; c < 3; ++c, ++n) {
      const double d = static_cast<double>(a[i + c]) - b[i + c];
      se += d * d;
    }
  return se == 0 ? std::numeric_limits<double>::infinity() : 10 * std::log10(255.0 * 255.0 / (se / static_cast<double>(n)));
}
}  // namespace

// (a) Camera 2.75 m from the box face x = -0.25, optical axis along +x through the face centre: the centre pixel's
// linear depth is exactly 2.75 m.
TEST(render, depth_center_matches_analytic) {
  Renderer r;
  std::cout << "device: " << r.device_name() << "\n";
  const auto s = scene();
  const Image im = r.render(*s, Camera{{-3, 0, 0.25}, {0, 0, 0.25}, 45}, 161, 121);  // odd size: a pixel sits on the axis
  const float d = im.depth[static_cast<std::size_t>(60) * 161 + 80];
  std::cout << "center depth " << d << " m, analytic 2.75, error " << std::abs(d - 2.75) << " m\n";
  EXPECT_NEAR(d, 2.75, 0.01);
  EXPECT_TRUE(std::isinf(im.depth[0]) || im.depth[0] > 2.75f);  // top-left corner looks over the floor/background
}

// The solver seam's mesh visuals: unrolled triangles whose stored normals agree with the winding (outward).
TEST(render, mesh_visuals_unrolled_with_outward_normals) {
  const auto s = scene();
  const std::vector<GeomVisual> vis = s->visuals();
  const GeomVisual* mesh = nullptr;
  for (const GeomVisual& v : vis)
    if (v.type == GeomType::mesh) mesh = &v;
  ASSERT_NE(mesh, nullptr);
  ASSERT_EQ(mesh->mesh.size(), 4u * 3 * 6);  // tetrahedron: 4 triangles x 3 vertices x (pos + normal)
  double c[3] = {0, 0, 0};  // centroid: every vertex appears 3 times
  for (std::size_t i = 0; i < 12; ++i)
    for (int k = 0; k < 3; ++k) c[k] += mesh->mesh[i * 6 + k] / 12.0;
  for (std::size_t t = 0; t < 4; ++t) {
    const float* v = mesh->mesh.data() + t * 18;
    double a[3], b[3], n[3];
    for (int k = 0; k < 3; ++k) {
      a[k] = v[6 + k] - v[k];
      b[k] = v[12 + k] - v[k];
    }
    n[0] = a[1] * b[2] - a[2] * b[1];
    n[1] = a[2] * b[0] - a[0] * b[2];
    n[2] = a[0] * b[1] - a[1] * b[0];
    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    EXPECT_GT(n[0] * (v[0] - c[0]) + n[1] * (v[1] - c[1]) + n[2] * (v[2] - c[2]), 0) << "triangle " << t << " winds inward";
    for (int i = 0; i < 3; ++i) {
      const float* nv = v + 6 * i + 3;
      const double d = (n[0] * nv[0] + n[1] * nv[1] + n[2] * nv[2]) / len;
      EXPECT_GT(d, 0.99) << "triangle " << t << " vertex " << i;
    }
  }
}

// (d) lidar vs analytic distances to the box (centre (0,0,0.25), half size 0.25) and the floor, to 1e-9; also through
// the bus (binary reply: full double precision).
TEST(render, lidar_matches_analytic_box) {
  const auto s = scene();
  const double o1[3] = {-2, 0, 0.25}, o2[3] = {0, 0, 3}, o3[3] = {2, 2, 1};
  const double a1 = std::atan2(0.1, 1);  // ray (1, 0.1, 0): hits the face x = -0.25 at t = 1.75 / cos(a)
  struct Case { const double* o; std::vector<double> d; double want; } cases[] = {
      {o1, {1, 0, 0}, 1.75},
      {o1, {2, 0, 0}, 1.75},  // length does not matter
      {o1, {1, 0.1, 0}, 1.75 / std::cos(a1)},
      {o2, {0, 0, -1}, 2.5},  // down onto the box top
      {o3, {0, 0, -1}, 1.0},  // down onto the floor
      {o1, {0, 0, 1}, -1.0},  // up into the sky: miss
  };
  double worst = 0;
  for (const Case& c : cases) {
    const std::vector<double> got = s->raycast({c.o, 3}, c.d);
    ASSERT_EQ(got.size(), 1u);
    worst = std::max(worst, std::abs(got[0] - c.want));
    EXPECT_NEAR(got[0], c.want, 1e-9);
  }
  std::cout << "lidar max error " << worst << " m\n";

  World w(load_mujoco, kScene);
  flatbuffers::FlatBufferBuilder fbb;
  const double origin[3] = {-2, 0, 0.25}, dirs[6] = {1, 0, 0, 1, 0.1, 0};
  fbb.Finish(dvsim::CreateEnvelope(fbb, 1, fbb.CreateString("test"), dvsim::Command::Raycast,
                                  dvsim::CreateRaycast(fbb, fbb.CreateVector(origin, 3), fbb.CreateVector(dirs, 6)).Union()));
  Session session;
  session.caps = {"read"};
  const flatbuffers::DetachedBuffer reply = w.execute({fbb.GetBufferPointer(), fbb.GetSize()}, session);
  const dvsim::Reply* rep = flatbuffers::GetRoot<dvsim::Reply>(reply.data());
  ASSERT_TRUE(rep->ok()) << rep->message()->str();
  ASSERT_EQ(rep->distances()->size(), 2u);
  EXPECT_NEAR((*rep->distances())[0], 1.75, 1e-9);
  EXPECT_NEAR((*rep->distances())[1], 1.75 / std::cos(a1), 1e-9);
}

// (e) error paths: bad size is invalid_argument (before any GPU work); with no Vulkan device the reply is a clear
// error, not a crash.
TEST(render, error_bad_size) {
  World w(load_mujoco, kScene);
  w.set_render_hook(make_render_hook());
  Session session;
  session.caps = {"read"};
  const std::string cam = R"("pos":[-3,0,1],"lookat":[0,0,0.25])";
  for (const char* size : {R"("width":0,"height":10)", R"("width":10,"height":4097)"}) {
    const std::string reply = execute_json(w, R"({"cmd_type":"Render","cmd":{)" + cam + "," + size + "}}", session);
    EXPECT_NE(reply.find(R"("ok": false)"), std::string::npos) << reply;
    EXPECT_NE(reply.find("invalid_argument"), std::string::npos) << reply;
  }
}

// The default device is a real GPU: it draws the scene within a tolerance of llvmpipe (a different rasteriser: edge / MSAA /
// shadow differences), and a CPU renderer is refused without DVSIM_RENDER_ALLOW_CPU=1. Compared with llvmpipe in the same
// process. (ctest runs these two without the CPU environment the render.* tests get, see CMakeLists.txt.)
TEST(render_gpu, smoke_default_device_is_not_cpu_and_matches_llvmpipe_loosely) {
  const auto s = scene();
  std::string n;
  Image a;
  {  // (one Renderer at a time: volk's function table is process-global)
    Renderer gpu;
    n = gpu.device_name();
    a = gpu.render(*s, kCam, kW, kH);
  }
  EXPECT_EQ(n.find("llvmpipe"), std::string::npos) << n;
  EXPECT_EQ(n.find("lavapipe"), std::string::npos) << n;
  ASSERT_EQ(setenv("DVSIM_RENDER_ALLOW_CPU", "1", 1), 0);  // own process: ctest runs one test each
  ASSERT_EQ(setenv("DVSIM_VK_DEVICE", "llvmpipe", 1), 0);
  Renderer cpu;
  const Image b = cpu.render(*s, kCam, kW, kH);
  const double p = psnr_rgb(a.rgba, b.rgba);
  std::cout << "device: " << n << " vs " << cpu.device_name() << "\nPSNR: " << p << " dB\n";
  EXPECT_GT(p, 28);
}

TEST(render_gpu, cpu_device_is_refused_unless_allowed) {
  ASSERT_EQ(unsetenv("DVSIM_RENDER_ALLOW_CPU"), 0);
  ASSERT_EQ(setenv("DVSIM_VK_DEVICE", "llvmpipe", 1), 0);  // own process: ctest runs one test each
  try {
    Renderer r;
    FAIL() << "a CPU renderer was used without DVSIM_RENDER_ALLOW_CPU";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("DVSIM_RENDER_ALLOW_CPU"), std::string::npos) << e.what();
  }
  ASSERT_EQ(setenv("DVSIM_RENDER_ALLOW_CPU", "1", 1), 0);
  Renderer r;
  EXPECT_NE(r.device_name().find("llvmpipe"), std::string::npos) << r.device_name();
}

// Weather look: rain=0 is the clear look bit for bit (and so is the default with no rain device); rain=50 changes the image (overcast sky,
// dim sun, haze, streaks) and darkens it, but not the linear depth.
TEST(render, rain_look_changes_with_the_rate_and_clear_is_unchanged) {
  const auto s = scene();
  Renderer r;
  const Image clear = r.render(*s, kCam, kW, kH);
  r.set_style(parse_render_style("rain=0"));
  const Image zero = r.render(*s, kCam, kW, kH);
  EXPECT_EQ(clear.rgba, zero.rgba);
  r.set_style(parse_render_style("rain=50"));
  const Image wet = r.render(*s, kCam, kW, kH);
  EXPECT_EQ(clear.depth, wet.depth);
  std::size_t diff = 0;
  for (std::size_t i = 0; i < clear.rgba.size(); i += 4)
    diff += clear.rgba[i] != wet.rgba[i] || clear.rgba[i + 1] != wet.rgba[i + 1] || clear.rgba[i + 2] != wet.rgba[i + 2];
  EXPECT_GT(diff, clear.rgba.size() / 4 / 2);
  EXPECT_THROW(parse_render_style("rain=-1"), std::invalid_argument);
  EXPECT_THROW(parse_render_style("overcast=2"), std::invalid_argument);
}

TEST(render, error_no_vulkan_device) {
  ASSERT_EQ(setenv("VK_DRIVER_FILES", "/nonexistent/none_icd.json", 1), 0);  // own process: ctest runs one test each
  try {
    Renderer r;
    FAIL() << "Renderer constructed without a driver";
  } catch (const std::runtime_error& e) {
    std::cout << "no-device error: " << e.what() << "\n";
    EXPECT_NE(std::string(e.what()).find("Vulkan"), std::string::npos);
  }
  World w(load_mujoco, kScene);
  w.set_render_hook(make_render_hook());
  Session session;
  session.caps = {"read"};
  const std::string reply = execute_json(
      w, R"({"cmd_type":"Render","cmd":{"pos":[-3,0,1],"lookat":[0,0,0.25],"width":32,"height":32}})", session);
  EXPECT_NE(reply.find(R"("ok": false)"), std::string::npos) << reply;
  EXPECT_NE(reply.find("solver_error"), std::string::npos) << reply;
  EXPECT_NE(reply.find("Vulkan"), std::string::npos) << reply;
}
