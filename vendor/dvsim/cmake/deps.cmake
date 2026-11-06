# All FetchContent pins live here. Tags verified on GitHub 2026-10-06.
include(FetchContent)

# MuJoCo 3.7.0 (Apache-2.0). Only the library: no simulate/samples/tests/python utils.
# Its own transitive deps (tinyxml2, qhull, lodepng, ccd, tinyobjloader, MarchingCubeCpp)
# are fetched by MuJoCo itself into the same _deps dir (pinned by its cmake/MujocoDependencies.cmake).
set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_SIMULATE OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_STUDIO OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MUJOCO_TEST_PYTHON_UTIL OFF CACHE BOOL "" FORCE)
set(MUJOCO_WITH_USD OFF CACHE BOOL "" FORCE)
set(MUJOCO_USE_FILAMENT OFF CACHE BOOL "" FORCE)
# MuJoCo auto-enables LTO and prefers lld; GCC's LTO objects through lld yield an EMPTY libmujoco.so
# (link "succeeds", symbols missing). Pre-seed its lld probe result so it falls back to the default linker.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(SUPPORTS_LLD OFF CACHE INTERNAL "")
endif()
FetchContent_Declare(mujoco
  GIT_REPOSITORY https://github.com/google-deepmind/mujoco.git
  GIT_TAG 3.7.0  # 72cb2b210da666617924de709406d6aadbe60c71
  GIT_SHALLOW TRUE)

# FlatBuffers v25.12.19 (Apache-2.0), target `flatc` is built for codegen in later milestones.
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG v25.12.19  # 282dcb1c3266b45600510da4810092f6ec4c85f2
  GIT_SHALLOW TRUE)

# GoogleTest v1.18.0 (BSD-3-Clause)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.18.0  # 063de7e9578f82b369302001269680b4b1553359
  GIT_SHALLOW TRUE)

# nanobind v3.1.0 (BSD-3-Clause; vendors robin_map, MIT, as a submodule). Declared here, made available
# in CMakeLists.txt only when DVSIM_PYTHON=ON (it needs Python found first).
FetchContent_Declare(nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v3.1.0  # 68480a9e6883bb1bf65e1925423321e2a89d07f9 (annotated tag; tag object 1852d88c)
  GIT_SHALLOW TRUE)

# Renderer deps (M9). Used as sources only (SOURCE_SUBDIR points at a directory without a CMakeLists.txt, so their own
# build files are never run): include dirs + our own targets in CMakeLists.txt. Target API is Vulkan 1.3; the headers are
# pinned to the 1.3.275 SDK = the loader version on the reference host (hfbox), volk loads everything dynamically.
# Vulkan-Headers (Apache-2.0 / MIT)
FetchContent_Declare(vulkan_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG vulkan-sdk-1.3.275.0  # 217e93c664ec6704ec2d8c36fa116c1a4a1e2d40
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR dvsim_sources_only)
# volk (MIT): meta-loader, no link-time libvulkan
FetchContent_Declare(volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG vulkan-sdk-1.3.275.0  # f2a16e3e19c2349b873343b2dc38a1d4c25af23a
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR dvsim_sources_only)
# Vulkan Memory Allocator v3.2.1 (MIT), header-only
FetchContent_Declare(vma
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG v3.2.1  # c788c52156f3ef7bc7ab769cb03c110a53ac8fcb
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR dvsim_sources_only)

# PX4 bridge deps (N5), header-only, used as sources only (our own INTERFACE targets are in CMakeLists.txt).
# mavlink/c_library_v2 (generated C headers, `common` dialect used): the repo has no tags and no LICENSE file, so it is
# pinned by commit (master of 2026-10-07). Licence: the generator repo's COPYING (https://github.com/mavlink/mavlink/blob/
# master/COPYING) says the generator is (L)GPL v3 but, "as an exception", a product embedding the generator OUTPUT may be
# redistributed under the MIT licence => the headers are MIT (docs/research/px4-sitl.md). The generator is not used.
FetchContent_Declare(mavlink
  GIT_REPOSITORY https://github.com/mavlink/c_library_v2.git
  GIT_TAG 56f6435ee725ed15d2e6d2a4c97ab6b41886ffc9  # master, no tags; full clone: a shallow clone cannot check out a bare commit
  SOURCE_SUBDIR dvsim_sources_only)
# toml++ v3.4.0 (MIT), header-only: bridge config
FetchContent_Declare(tomlplusplus
  GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
  GIT_TAG v3.4.0  # 30172438cee64926dc41fdd9c11fb3ba5b2ba9de (lightweight tag)
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR dvsim_sources_only)

# Dog whole-body controller in C++ (examples/dog/loco/cpp, option DVSIM_LOCO_CPP, SP4 Task 7 Step 6). Declared only: made available in
# that directory's CMakeLists.txt, so a normal build never fetches them.
# Eigen 3.4.0 (MPL-2.0; header-only, used as sources: SOURCE_SUBDIR keeps its own tests/docs out of the build)
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0  # 3147391d946bb4b6c68edd901f2add6ac1f31f8c
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR dvsim_sources_only)
# OSQP C library v1.0.0 (Apache-2.0), builtin algebra (QDLDL, Apache-2.0, fetched by OSQP itself); the QP solver of the Python wheel too
FetchContent_Declare(osqp
  GIT_REPOSITORY https://github.com/osqp/osqp.git
  GIT_TAG v1.0.0  # 236713ce9a56c182ac3230d52108f952afce1523
  GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(mujoco flatbuffers googletest vulkan_headers volk vma mavlink tomlplusplus)

# Viewer deps (M10, Linux/X11 only for now: the Windows/macOS GLFW source lists are not written or tested). Sources only,
# like the renderer deps: our own targets are in CMakeLists.txt.
if(DVSIM_VIEWER)
  # GLFW 3.4 (zlib). Wayland is not built. GLFW dlopen()s libX11 (required) and libXrandr/libXi/libXext/libXrender/
  # libXxf86vm/libXinerama/libXcursor/libX11-xcb (each optional, checked at run time), so the build needs X11 HEADERS only.
  FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4  # 7b6aead9fb88b3623e3b3725ebb42670cbe4c579 (annotated tag; tag object a74efa0d)
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR dvsim_sources_only)
  # Dear ImGui v1.92.9b (MIT): core + the GLFW and Vulkan backends (volk, dynamic rendering).
  FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9b  # f1cc2ae15e53a861a874c3034aae6798fde194ab (lightweight tag)
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR dvsim_sources_only)
  # X11 compile-time headers (MIT/X11 licences), fetched shallow from freedesktop.org instead of a system -dev package.
  # Only include/ is used; the repos' library sources are never built. Needed by GLFW's x11_*.c (tags verified 2026-10-06):
  #   xorgproto 2024.1  X.h Xproto.h Xfuncproto.h keysym.h XKB.h randr/render/XI2/shape protocol headers ...
  #   libX11 1.8.12     Xlib.h Xutil.h Xresource.h XKBlib.h cursorfont.h
  #   libXrandr 1.5.4, libXrender 0.9.12, libXi 1.8.2, libXinerama 1.1.5, libXcursor 1.2.3, libXext 1.3.7 (Xge.h),
  #   libXfixes 6.0.1 (Xfixes.h, pulled in by XInput2.h)
  # Odd historical notice: libX11's src/xlibi18n/lcUniConv/COPYRIGHT says those (never compiled here) files derive from
  # LGPL libiconv and were relicensed MIT-style by the FSF; check_licenses.py reports it as a NOTE, not a failure.
  foreach(_x IN ITEMS
      "xorgproto|proto/xorgproto|xorgproto-2024.1|67469711055522b8adb2d795b01e7ba98cb8816c"
      "libx11|lib/libx11|libX11-1.8.12|59917d28a3c41ad22d6fc52e323cafe2cdd596d5"
      "libxrandr|lib/libxrandr|libXrandr-1.5.4|512bf0b15b5597c721ff8c61083616ca9040fa72"
      "libxrender|lib/libxrender|libXrender-0.9.12|46e12ff9e8e4d3f0313a2f097df93dbfdc14f92f"
      "libxi|lib/libxi|libXi-1.8.2|3c2d5aedd7708c75710b39268631762de904c304"
      "libxinerama|lib/libxinerama|libXinerama-1.1.5|71dfee64feb84f907016940263b235a61c9e8960"
      "libxcursor|lib/libxcursor|libXcursor-1.2.3|8ae6cf61b18ecddf26f72a07bbfbd1ea9f022c36"
      "libxext|lib/libxext|libXext-1.3.7|2a694ba264ccdb205909909abfe8b136f9156ebe"
      "libxfixes|lib/libxfixes|libXfixes-6.0.1|c1cab28e27dd1c5a81394965248b57e490ccf2ca")
    string(REPLACE "|" ";" _x "${_x}")
    list(GET _x 0 _name)
    list(GET _x 1 _path)
    list(GET _x 2 _tag)
    FetchContent_Declare(${_name}
      GIT_REPOSITORY https://gitlab.freedesktop.org/xorg/${_path}.git
      GIT_TAG ${_tag}  # commit: see the list above
      GIT_SHALLOW TRUE
      SOURCE_SUBDIR dvsim_sources_only)
    list(APPEND _x11_repos ${_name})
  endforeach()
  # stb_image v2.30 (public domain / MIT, dual-licensed in the header), ONE header: decodes the PNG part thumbnails of the
  # library page. nothings/stb has no tags, so pinned by commit + SHA256 of the file (verified 2026-10-07).
  FetchContent_Declare(stb_image
    URL https://raw.githubusercontent.com/nothings/stb/2c980bb59875b0d32144a71867fbdebb2f77cd20/stb_image.h
    URL_HASH SHA256=594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3
    DOWNLOAD_NO_EXTRACT TRUE)
  # The UI font is Roboto-Medium.ttf from Dear ImGui's own misc/fonts (Apache-2.0, ships in the pinned imgui checkout above);
  # CMakeLists.txt embeds it into the viewer as a byte array at configure time. No further dependency.
  # ImGuiColorTextEdit v1.92.9 (MIT, goossens' rewrite of Balazs Jako's editor, built for Dear ImGui 1.92.9): the PROGRAM page's
  # code editor (syntax colouring, find / replace, multi-cursor, bracket matching, undo). TextEditor.cpp/.h are compiled, sources
  # only. Tag v1.92.9 = commit 281be7cdc504ad789afa020dfae61844208addeb, pinned by archive SHA256 (verified 2026-10-12).
  FetchContent_Declare(imguicolortextedit
    URL https://github.com/goossens/ImGuiColorTextEdit/archive/281be7cdc504ad789afa020dfae61844208addeb.tar.gz
    URL_HASH SHA256=9a193778351d6ba7d789639f1ec049fca82ef4c88161133ffe5f14861a022e79
    SOURCE_SUBDIR dvsim_sources_only)
  FetchContent_MakeAvailable(glfw imgui imguicolortextedit stb_image ${_x11_repos})
endif()

# slangc 2026.19 (Apache-2.0 WITH LLVM-exception): prebuilt, BUILD-TIME TOOL ONLY (never linked or shipped). The release
# archive is ~30 MB / ~200 MB unpacked, so it is downloaded to a scratch dir with a pinned SHA256, only slangc and the
# libraries it loads are extracted (plus the license files), and the archive is deleted. Result: build/_deps/slang-bin/.
# Override with -DDVSIM_SLANGC=/path/to/slangc. Pins verified 2026-10-06; only the linux-x86_64 one has been run
# (hfbox); the other layouts come from the archive listings.
set(DVSIM_SLANGC "" CACHE FILEPATH "slangc to use instead of the pinned download")
if(NOT DVSIM_SLANGC OR NOT EXISTS "${DVSIM_SLANGC}")
  set(_slang_ver 2026.19)
  set(_slang_url https://github.com/shader-slang/slang/releases/download/v${_slang_ver})
  if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(_slang_archive slang-${_slang_ver}-linux-x86_64-glibc-2.28.tar.gz)
    set(_slang_sha256 bd7afcb0956dee9304f44ed115e9aaa7140704271b5c3294bcb237e3ae67adc7)
    set(_slang_exe bin/slangc)
    set(_slang_keep bin/slangc "lib/libslang-compiler.so*" LICENSE "LICENSES/*")
  elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_slang_archive slang-${_slang_ver}-linux-aarch64-glibc-2.28.tar.gz)
    set(_slang_sha256 f49229eb9606b47b122e1d8789be46a7bcd6bb2ffa94b06fae6d544a4e6df637)
    set(_slang_exe bin/slangc)
    set(_slang_keep bin/slangc "lib/libslang-compiler.so*" LICENSE "LICENSES/*")
  elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(_slang_archive slang-${_slang_ver}-windows-x86_64.zip)
    set(_slang_sha256 fc922f214b5d115f663932e59e80e0f6b6cc6fae140979cf307da6459b6f8f33)
    set(_slang_exe bin/slangc.exe)
    set(_slang_keep bin/slangc.exe bin/slang-compiler.dll LICENSE "LICENSES/*")
  else()
    message(FATAL_ERROR "No pinned slangc for ${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_HOST_SYSTEM_PROCESSOR}: "
                        "pass -DDVSIM_SLANGC=/path/to/slangc or add a pin in cmake/deps.cmake")
  endif()
  set(_slang_dir ${FETCHCONTENT_BASE_DIR}/slang-bin)
  if(NOT EXISTS ${_slang_dir}/${_slang_exe})
    set(_slang_tmp ${FETCHCONTENT_BASE_DIR}/slang-download)
    file(REMOVE_RECURSE ${_slang_tmp})
    message(STATUS "Fetching slangc ${_slang_ver} (${_slang_archive})")
    file(DOWNLOAD ${_slang_url}/${_slang_archive} ${_slang_tmp}/${_slang_archive}
         EXPECTED_HASH SHA256=${_slang_sha256} STATUS _slang_st)
    list(GET _slang_st 0 _slang_rc)
    if(_slang_rc EQUAL 0)
      file(ARCHIVE_EXTRACT INPUT ${_slang_tmp}/${_slang_archive} DESTINATION ${_slang_dir} PATTERNS ${_slang_keep})
    endif()
    file(REMOVE_RECURSE ${_slang_tmp})  # always: the archive must not stay in _deps
    if(NOT _slang_rc EQUAL 0 OR NOT EXISTS ${_slang_dir}/${_slang_exe})
      message(FATAL_ERROR "slangc fetch failed: ${_slang_st}")
    endif()
  endif()
  set(DVSIM_SLANGC ${_slang_dir}/${_slang_exe} CACHE FILEPATH "slangc to use instead of the pinned download" FORCE)
endif()

# Risk 2: mju_user_error handler throws a C++ exception through MuJoCo's C frames -> C needs unwind tables.
if(NOT MSVC)
  target_compile_options(mujoco PRIVATE $<$<COMPILE_LANGUAGE:C>:-fexceptions>)
endif()

# MuJoCo 3.7.0 hardcodes -Werror; GCC 16 + new glibc const-correct strchr() trips
# -Wdiscarded-qualifiers in engine_print.c. Demote that one warning (appended after MuJoCo's -Werror).
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  target_compile_options(mujoco PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wno-error=discarded-qualifiers>)
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")  # clang names it differently
  target_compile_options(mujoco PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wno-error=incompatible-pointer-types-discards-qualifiers>)
endif()
