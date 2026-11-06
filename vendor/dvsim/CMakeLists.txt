cmake_minimum_required(VERSION 3.25)
project(dvsim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler cache (sccache, Apache-2.0) when it is on PATH; laptop/CI without it are unaffected. -DDVSIM_SCCACHE=OFF disables.
# Set before deps so MuJoCo/FlatBuffers/GLFW objects are cached too. tools/remote.sh points it at the shared cache.
option(DVSIM_SCCACHE "Use sccache as compiler launcher when found" ON)
find_program(DVSIM_SCCACHE_EXE sccache)
if(DVSIM_SCCACHE AND DVSIM_SCCACHE_EXE)
  # Never reuse a stale object after a header change (a src/core/solver.h edit once served old objects -> segfaults): sccache's
  # direct mode keys on a manifest of the headers of an EARLIER compile of the same source, and one shared server (config read at
  # its start) served every worktree. SCCACHE_DIRECT=false = always run the preprocessor and key on its output, which cannot be
  # stale. Each checkout gets its own server (a port from the source dir; tools/remote.sh starts the same one) so a server another
  # checkout started in direct mode is never reused.
  string(MD5 _sc_hash "${CMAKE_SOURCE_DIR}")
  string(SUBSTRING "${_sc_hash}" 0 2 _sc_hex)
  math(EXPR _sc_port "4300 + 0x${_sc_hex}")  # 4300..4555
  set(_sc_launcher ${CMAKE_COMMAND} -E env SCCACHE_DIRECT=false SCCACHE_SERVER_PORT=${_sc_port} ${DVSIM_SCCACHE_EXE})
  set(CMAKE_C_COMPILER_LAUNCHER ${_sc_launcher})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${_sc_launcher})
endif()

# Determinism (plan Risk 1): no FMA contraction, no fast-math. Set before deps so they inherit it.
if(NOT MSVC)
  add_compile_options(-ffp-contract=off)
endif()

# dvsim-view / Sim.show(): GLFW + Dear ImGui. Linux (X11) only until the Windows/macOS GLFW build is written and tested.
if(UNIX AND NOT APPLE)
  option(DVSIM_VIEWER "Build the Vulkan viewer (dvsim-view)" ON)
else()
  option(DVSIM_VIEWER "Build the Vulkan viewer (dvsim-view)" OFF)
endif()

include(cmake/deps.cmake)

enable_testing()
include(GoogleTest)

# dvsim_schema: dvsim.fbs -> C++ header, embedded .bfbs header (runtime reflection/JSON), JSON Schema (MCP, M8).
# Re-runs whenever schema/dvsim.fbs or flatc changes.
set(DVSIM_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
add_custom_command(
  OUTPUT ${DVSIM_GEN_DIR}/dvsim_generated.h ${DVSIM_GEN_DIR}/dvsim_bfbs_generated.h ${DVSIM_GEN_DIR}/dvsim.schema.json
  COMMAND flatc --cpp --scoped-enums --bfbs-gen-embed --jsonschema -o ${DVSIM_GEN_DIR} ${CMAKE_SOURCE_DIR}/schema/dvsim.fbs
  DEPENDS flatc ${CMAKE_SOURCE_DIR}/schema/dvsim.fbs
  COMMENT "flatc: dvsim.fbs")
# devices.fbs (N2): device config for MujocoSolver, separate from dvsim.fbs (whose hash is in replay logs).
add_custom_command(
  OUTPUT ${DVSIM_GEN_DIR}/devices_generated.h ${DVSIM_GEN_DIR}/devices_bfbs_generated.h
  COMMAND flatc --cpp --scoped-enums --bfbs-gen-embed -o ${DVSIM_GEN_DIR} ${CMAKE_SOURCE_DIR}/schema/devices.fbs
  DEPENDS flatc ${CMAKE_SOURCE_DIR}/schema/devices.fbs
  COMMENT "flatc: devices.fbs")
add_custom_target(dvsim_schema_gen DEPENDS ${DVSIM_GEN_DIR}/dvsim_generated.h ${DVSIM_GEN_DIR}/devices_generated.h)
add_library(dvsim_schema INTERFACE)
add_dependencies(dvsim_schema dvsim_schema_gen)
target_include_directories(dvsim_schema INTERFACE ${DVSIM_GEN_DIR})
target_link_libraries(dvsim_schema INTERFACE flatbuffers)

# Seam enforced by the build graph: dvsim_core gets no mujoco include dirs, so including mujoco.h fails.
find_package(Threads REQUIRED)
add_library(dvsim_core STATIC src/core/world.cpp src/core/bus.cpp src/core/replay.cpp src/core/host.cpp
  src/core/monitor.cpp src/core/shm.cpp)
target_include_directories(dvsim_core PUBLIC src include)  # include/: dvsim/shm_client.hpp, the dependency-free shm client
target_link_libraries(dvsim_core PUBLIC dvsim_schema Threads::Threads)

add_library(dvsim_mujoco STATIC src/mujoco/mujoco_solver.cpp src/mujoco/devices/devices.cpp
  src/mujoco/devices/electrical.cpp src/mujoco/devices/aero.cpp src/mujoco/devices/servo.cpp
  src/mujoco/devices/sensors.cpp src/mujoco/devices/launcher.cpp src/mujoco/devices/rocket.cpp
  src/mujoco/devices/vessel.cpp src/mujoco/devices/rangefinder.cpp src/mujoco/devices/vane.cpp
  src/mujoco/devices/windfield.cpp
  src/mujoco/devices/rain.cpp src/mujoco/devices/qdd.cpp src/mujoco/devices/contact.cpp)
target_link_libraries(dvsim_mujoco PUBLIC dvsim_core mujoco)

# Static libs end up inside the _dvsim shared module: they need -fPIC.
set_target_properties(dvsim_core dvsim_mujoco flatbuffers lodepng PROPERTIES POSITION_INDEPENDENT_CODE ON)

# dvsim_render: headless Vulkan 1.3 renderer (no window system). dvsim_core never links it: the app installs it into a
# World as a RenderHook (src/core/render.h). volk loads libvulkan at run time, so binaries start without a GPU.
include(cmake/shaders.cmake)
add_library(dvsim_volk STATIC ${volk_SOURCE_DIR}/volk.c)
target_include_directories(dvsim_volk PUBLIC ${volk_SOURCE_DIR} ${vulkan_headers_SOURCE_DIR}/include)
target_compile_definitions(dvsim_volk PUBLIC VK_NO_PROTOTYPES)
target_link_libraries(dvsim_volk PUBLIC ${CMAKE_DL_LIBS})
dvsim_embed_shader(geom)
add_library(dvsim_render STATIC src/render/renderer.cpp ${DVSIM_GEN_DIR}/shaders/geom_spv.h)
target_include_directories(dvsim_render PUBLIC src PRIVATE ${vma_SOURCE_DIR}/include ${DVSIM_GEN_DIR}/shaders)
target_link_libraries(dvsim_render PUBLIC dvsim_core PRIVATE dvsim_volk)
set_target_properties(dvsim_render dvsim_volk PROPERTIES POSITION_INDEPENDENT_CODE ON)  # linked into _dvsim.so

# dvsim_viewer (M10): GLFW window + Vulkan swapchain + Dear ImGui over a Host. dvsim-view is its command line; _dvsim links it
# for Sim.show(). GLFW and ImGui are built from fetched sources (no find_package): GLFW dlopen()s libX11 & co, so only the
# X11 headers (fetched, cmake/deps.cmake) are needed to compile it.
if(DVSIM_VIEWER)
  set(_x11_gen ${CMAKE_BINARY_DIR}/x11-gen)  # the one generated X11 header: Xcursor.h from its (unsubstituted) .in
  configure_file(${libxcursor_SOURCE_DIR}/include/X11/Xcursor/Xcursor.h.in ${_x11_gen}/X11/Xcursor/Xcursor.h COPYONLY)
  set(_glfw_src ${glfw_SOURCE_DIR}/src)
  add_library(dvsim_glfw STATIC
    ${_glfw_src}/context.c ${_glfw_src}/init.c ${_glfw_src}/input.c ${_glfw_src}/monitor.c ${_glfw_src}/platform.c
    ${_glfw_src}/vulkan.c ${_glfw_src}/window.c ${_glfw_src}/egl_context.c ${_glfw_src}/osmesa_context.c
    ${_glfw_src}/null_init.c ${_glfw_src}/null_monitor.c ${_glfw_src}/null_window.c ${_glfw_src}/null_joystick.c
    ${_glfw_src}/posix_time.c ${_glfw_src}/posix_thread.c ${_glfw_src}/posix_module.c ${_glfw_src}/posix_poll.c
    ${_glfw_src}/linux_joystick.c ${_glfw_src}/x11_init.c ${_glfw_src}/x11_monitor.c ${_glfw_src}/x11_window.c
    ${_glfw_src}/xkb_unicode.c ${_glfw_src}/glx_context.c)
  target_compile_definitions(dvsim_glfw PRIVATE _GLFW_X11 INTERFACE GLFW_INCLUDE_NONE)  # NONE (for users, not GLFW itself): no GL headers; Vulkan comes from volk
  target_include_directories(dvsim_glfw PUBLIC ${glfw_SOURCE_DIR}/include)
  target_include_directories(dvsim_glfw SYSTEM PRIVATE ${_x11_gen}
    ${xorgproto_SOURCE_DIR}/include ${libx11_SOURCE_DIR}/include ${libxrandr_SOURCE_DIR}/include
    ${libxrender_SOURCE_DIR}/include ${libxi_SOURCE_DIR}/include ${libxinerama_SOURCE_DIR}/include
    ${libxext_SOURCE_DIR}/include ${libxfixes_SOURCE_DIR}/include)
  target_link_libraries(dvsim_glfw PUBLIC ${CMAKE_DL_LIBS} Threads::Threads m)

  set(_imgui_src ${imgui_SOURCE_DIR})
  add_library(dvsim_imgui STATIC ${_imgui_src}/imgui.cpp ${_imgui_src}/imgui_draw.cpp ${_imgui_src}/imgui_tables.cpp
    ${_imgui_src}/imgui_widgets.cpp ${_imgui_src}/backends/imgui_impl_glfw.cpp ${_imgui_src}/backends/imgui_impl_vulkan.cpp)
  target_include_directories(dvsim_imgui SYSTEM PUBLIC ${_imgui_src} ${_imgui_src}/backends)
  target_compile_definitions(dvsim_imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
  target_link_libraries(dvsim_imgui PUBLIC dvsim_volk dvsim_glfw)

  # UI font: Roboto-Medium.ttf (Apache-2.0, from the pinned imgui checkout) as a byte array, rewritten only when it changes.
  file(READ ${_imgui_src}/misc/fonts/Roboto-Medium.ttf _font_hex HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _font_hex "${_font_hex}")
  file(READ ${_imgui_src}/misc/fonts/Cousine-Regular.ttf _mono_hex HEX)  # Cousine (Apache-2.0): the log / code view
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _mono_hex "${_mono_hex}")
  file(CONFIGURE OUTPUT ${DVSIM_GEN_DIR}/ui_font.cpp @ONLY CONTENT
    "#include <cstddef>\nextern unsigned char kUiFont[] = {@_font_hex@};\nextern const std::size_t kUiFontSize = sizeof kUiFont;\nextern unsigned char kUiMono[] = {@_mono_hex@};\nextern const std::size_t kUiMonoSize = sizeof kUiMono;\n")
  add_library(dvsim_viewer STATIC src/viewer/viewer.cpp src/viewer/theme.cpp src/viewer/build3d.cpp src/viewer/library_ui.cpp
    src/viewer/hud.cpp src/viewer/program_ui.cpp src/viewer/run_ui.cpp ${imguicolortextedit_SOURCE_DIR}/TextEditor.cpp ${DVSIM_GEN_DIR}/ui_font.cpp)
  target_include_directories(dvsim_viewer PRIVATE ${stb_image_SOURCE_DIR})
  target_include_directories(dvsim_viewer SYSTEM PRIVATE ${imguicolortextedit_SOURCE_DIR})  # third party: its warnings are not ours
  target_sources(dvsim_viewer PRIVATE src/viewer/wire2d.cpp)  # the WIRING page's picker and 2D canvas
  target_link_libraries(dvsim_viewer PUBLIC dvsim_render dvsim_core PRIVATE dvsim_imgui dvsim_glfw dvsim_volk)
  set_target_properties(dvsim_glfw dvsim_imgui dvsim_viewer PROPERTIES POSITION_INDEPENDENT_CODE ON)  # linked into _dvsim.so

  add_executable(dvsim_view src/viewer/main.cpp)
  set_target_properties(dvsim_view PROPERTIES OUTPUT_NAME dvsim-view)
  target_link_libraries(dvsim_view PRIVATE dvsim_viewer dvsim_mujoco mujoco)
endif()

add_executable(dvsim_cli src/cli/main.cpp)
set_target_properties(dvsim_cli PROPERTIES OUTPUT_NAME dvsim-replay)  # the Python CLI owns `dvsim`
target_link_libraries(dvsim_cli PRIVATE dvsim_core dvsim_mujoco mujoco)

add_executable(dvsim_shm_serve src/cli/shm_serve.cpp)  # shm lockstep server (M11); the client is include/dvsim/shm_client.hpp
set_target_properties(dvsim_shm_serve PROPERTIES OUTPUT_NAME dvsim-shm-serve)
target_link_libraries(dvsim_shm_serve PRIVATE dvsim_core dvsim_mujoco mujoco)

# dvsim-px4-bridge (N5): PX4 SITL lockstep bridge. Header-only deps (mavlink: SYSTEM, its generated code is not ours to lint).
add_library(dvsim_px4_deps INTERFACE)
target_include_directories(dvsim_px4_deps SYSTEM INTERFACE ${mavlink_SOURCE_DIR} ${tomlplusplus_SOURCE_DIR}/include)
add_executable(dvsim_px4_bridge src/px4/main.cpp)
set_target_properties(dvsim_px4_bridge PROPERTIES OUTPUT_NAME dvsim-px4-bridge)
target_link_libraries(dvsim_px4_bridge PRIVATE dvsim_core dvsim_mujoco mujoco dvsim_px4_deps)

# The shm test's child process: header-only client only, so it links nothing of dvsim (proves the header stands alone).
add_executable(dvsim_shm_test_client tests/cpp/shm_test_client.cpp)
target_include_directories(dvsim_shm_test_client PRIVATE include tests/cpp)

add_executable(dvsim_tests tests/cpp/test_smoke.cpp tests/cpp/test_determinism.cpp tests/cpp/test_bus.cpp
  tests/cpp/test_replay.cpp tests/cpp/test_host.cpp tests/cpp/test_shm.cpp tests/cpp/test_devices.cpp
  tests/cpp/test_launcher.cpp tests/cpp/test_rocket.cpp tests/cpp/test_vessel.cpp tests/cpp/test_rangefinder.cpp
  tests/cpp/test_jetvane.cpp tests/cpp/test_px4.cpp tests/cpp/test_terrain.cpp tests/cpp/test_rain.cpp
  tests/cpp/test_qdd.cpp tests/cpp/test_foot_contact.cpp tests/cpp/test_rosette.cpp)
add_dependencies(dvsim_tests dvsim_cli dvsim_shm_test_client dvsim_px4_bridge)  # test_replay / test_shm / test_px4 run these executables
target_include_directories(dvsim_tests PRIVATE tests/cpp)
target_link_libraries(dvsim_tests PRIVATE dvsim_core dvsim_mujoco mujoco dvsim_px4_deps GTest::gtest_main)
target_compile_definitions(dvsim_tests PRIVATE
  DVSIM_CLI_EXE="$<TARGET_FILE:dvsim_cli>"
  DVSIM_PX4_BRIDGE_EXE="$<TARGET_FILE:dvsim_px4_bridge>"
  DVSIM_SHM_CLIENT_EXE="$<TARGET_FILE:dvsim_shm_test_client>"
  DVSIM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
  DVSIM_HUMANOID_XML="${mujoco_SOURCE_DIR}/model/humanoid/humanoid.xml")
if(WIN32)  # no rpath on Windows: put mujoco.dll next to the test exe
  add_custom_command(TARGET dvsim_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mujoco> $<TARGET_FILE_DIR:dvsim_tests>)
  add_custom_command(TARGET dvsim_cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mujoco> $<TARGET_FILE_DIR:dvsim_cli>)
endif()
gtest_discover_tests(dvsim_tests PROPERTIES LABELS cpu)

# GPU tests: their own executable and ctest label, so `ctest --preset dev` (cpu) never touches a GPU.
# Run them with `ctest --preset gpu` (see CMakePresets.json); pick the driver with VK_DRIVER_FILES.
add_executable(dvsim_render_tests tests/cpp/test_render.cpp)
target_link_libraries(dvsim_render_tests PRIVATE dvsim_render dvsim_mujoco mujoco GTest::gtest_main)
target_compile_definitions(dvsim_render_tests PRIVATE DVSIM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
if(WIN32)
  add_custom_command(TARGET dvsim_render_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mujoco> $<TARGET_FILE_DIR:dvsim_render_tests>)
endif()
# render.* run on llvmpipe (deterministic; a CPU renderer is otherwise refused); render_gpu.* run on the default device (the GPU) with a tolerance.
gtest_discover_tests(dvsim_render_tests TEST_FILTER render.* PROPERTIES LABELS gpu
  ENVIRONMENT "DVSIM_RENDER_ALLOW_CPU=1;DVSIM_VK_DEVICE=llvmpipe")
gtest_discover_tests(dvsim_render_tests TEST_FILTER render_gpu.* PROPERTIES LABELS gpu)

if(DVSIM_VIEWER)  # needs a window system too: run under a display or `xvfb-run -a ctest --preset gpu`
  add_executable(dvsim_viewer_tests tests/cpp/test_viewer.cpp)
  add_dependencies(dvsim_viewer_tests dvsim_view dvsim_cli)  # the tests run the dvsim-view / dvsim-replay executables
  target_link_libraries(dvsim_viewer_tests PRIVATE dvsim_viewer dvsim_mujoco mujoco GTest::gtest_main)
  target_compile_definitions(dvsim_viewer_tests PRIVATE
    DVSIM_VIEW_EXE="$<TARGET_FILE:dvsim_view>"
    DVSIM_CLI_EXE="$<TARGET_FILE:dvsim_cli>"
    DVSIM_VIEWER_OUT_DIR="${CMAKE_BINARY_DIR}/viewer-out"
    DVSIM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  gtest_discover_tests(dvsim_viewer_tests PROPERTIES LABELS gpu)
endif()

# _dvsim: nanobind extension (CPython >= 3.12 stable ABI). Off by default; scikit-build-core turns it on
# (pyproject.toml) and shares this build tree, so the dependency sources/objects exist exactly once.
option(DVSIM_PYTHON "Build the _dvsim Python extension" OFF)
if(DVSIM_PYTHON)
  # scikit-build-core passes its temporary isolated-build interpreter, which is gone after `pip install`; the next
  # plain `cmake --build` regenerate would then fail on it ("Cannot run the interpreter"). Search again instead.
  if(DEFINED CACHE{Python_EXECUTABLE} AND NOT EXISTS "${Python_EXECUTABLE}")
    unset(Python_EXECUTABLE CACHE)
    unset(_Python_EXECUTABLE CACHE)
  endif()
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
  FetchContent_MakeAvailable(nanobind)
  # The flatc JSON Schema (MCP tool input schemas) is embedded in the module as `_dvsim.schema_json`.
  add_custom_command(
    OUTPUT ${DVSIM_GEN_DIR}/dvsim_schema_json.h
    COMMAND ${CMAKE_COMMAND} -DIN=${DVSIM_GEN_DIR}/dvsim.schema.json -DOUT=${DVSIM_GEN_DIR}/dvsim_schema_json.h
            -DNAME=kSchemaJson -P ${CMAKE_SOURCE_DIR}/cmake/embed_text.cmake
    DEPENDS ${DVSIM_GEN_DIR}/dvsim.schema.json ${CMAKE_SOURCE_DIR}/cmake/embed_text.cmake)
  nanobind_add_module(_dvsim STABLE_ABI src/python/bindings.cpp ${DVSIM_GEN_DIR}/dvsim_schema_json.h)
  target_link_libraries(_dvsim PRIVATE dvsim_core dvsim_mujoco mujoco dvsim_render lodepng)
  if(DVSIM_VIEWER)
    target_link_libraries(_dvsim PRIVATE dvsim_viewer)
    target_compile_definitions(_dvsim PRIVATE DVSIM_HAS_VIEWER)
  endif()
  set_target_properties(_dvsim PROPERTIES INSTALL_RPATH "$ORIGIN")  # libmujoco.so is installed next to it
  install(TARGETS _dvsim mujoco LIBRARY DESTINATION dvsim COMPONENT dvsim_python)
endif()

# loco_wbc_cpp: the dog's whole-body controller in C++ (SP4 Task 7 Step 6), a small nanobind module for examples/dog/loco/wbc.py to load.
# Off by default; not part of the installed dvsim package. Output: build/loco/.
option(DVSIM_LOCO_CPP "Build examples/dog/loco/cpp (the C++ whole-body controller)" OFF)
if(DVSIM_LOCO_CPP)
  # the repo venv's interpreter when the caller named none (a fresh build dir otherwise fails with "Could NOT find Python (missing: Interpreter)")
  if(NOT DEFINED Python_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
  endif()
  add_subdirectory(examples/dog/loco/cpp)
endif()
