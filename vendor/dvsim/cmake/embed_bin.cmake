# cmake -DIN=<binary file> -DOUT=<header> -DNAME=<identifier> -P embed_bin.cmake
# Writes `alignas(4) inline constexpr unsigned char NAME[] = {...};` (used for the SPIR-V shaders).
file(READ "${IN}" hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
file(WRITE "${OUT}" "alignas(4) inline constexpr unsigned char ${NAME}[] = {${bytes}};\n")
