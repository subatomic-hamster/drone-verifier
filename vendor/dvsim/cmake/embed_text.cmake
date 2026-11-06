# cmake -DIN=<text file> -DOUT=<header> -DNAME=<identifier> -P embed_text.cmake
# Writes `inline constexpr const char* NAME = R"DVSIM(<contents>)DVSIM";` (used for the flatc JSON Schema).
file(READ "${IN}" content)
file(WRITE "${OUT}" "inline constexpr const char* ${NAME} = R\"DVSIM(${content})DVSIM\";\n")
