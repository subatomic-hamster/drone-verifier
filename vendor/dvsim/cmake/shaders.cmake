# Slang -> SPIR-V -> (spirv-val) -> C++ header.
#   dvsim_embed_shader(<name>)   src/render/shaders/<name>.slang  =>  ${DVSIM_GEN_DIR}/shaders/<name>_spv.h
# The header defines `kSpv_<name>` (alignas(4) unsigned char[]). DVSIM_SLANGC comes from cmake/deps.cmake (pinned download).
# The SPIR-V is validated against the Vulkan 1.3 environment with the system spirv-val when it is installed (a missing
# spirv-val is reported at configure time and the validation step is skipped).
find_program(DVSIM_SPIRV_VAL spirv-val)
if(NOT DVSIM_SPIRV_VAL)
  message(STATUS "spirv-val not found: generated SPIR-V will not be validated")
endif()

function(dvsim_embed_shader name)
  set(src ${CMAKE_SOURCE_DIR}/src/render/shaders/${name}.slang)
  set(dir ${DVSIM_GEN_DIR}/shaders)
  if(DVSIM_SPIRV_VAL)
    set(validate COMMAND ${DVSIM_SPIRV_VAL} --target-env vulkan1.3 ${dir}/${name}.spv)
  else()
    set(validate "")
  endif()
  add_custom_command(
    OUTPUT ${dir}/${name}_spv.h
    COMMAND ${CMAKE_COMMAND} -E make_directory ${dir}
    # -O0: slangc's optimiser (spirv-opt) lives in libslang-glslang, which we deliberately do not extract (its notice
    # file carries glslang's vestigial GPL/Bison text); the plain SPIR-V is validated below and fine for these shaders.
    COMMAND ${DVSIM_SLANGC} ${src} -target spirv -O0 -fvk-use-entrypoint-name
            -entry vsMain -entry fsMain -entry fsLin -entry vsShadow -entry vsSky -entry fsSky -entry vsEdge -entry fsEdge -entry vsRain -entry fsRain
            -o ${dir}/${name}.spv
    ${validate}
    COMMAND ${CMAKE_COMMAND} -DIN=${dir}/${name}.spv -DOUT=${dir}/${name}_spv.h -DNAME=kSpv_${name}
            -P ${CMAKE_SOURCE_DIR}/cmake/embed_bin.cmake
    DEPENDS ${src} ${CMAKE_SOURCE_DIR}/cmake/embed_bin.cmake
    COMMENT "slangc + spirv-val: ${name}.slang"
    VERBATIM)
endfunction()
