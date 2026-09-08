from pathlib import Path


Import("env")

library_path = (
    Path(env.subst("$PROJECT_LIBDEPS_DIR"))
    / env.subst("$PIOENV")
    / "Arduino_BMI270_BMM150"
    / "src"
    / "BMI270.cpp"
)

source = library_path.read_text()
include = '#include "mbed.h"\n'

if include not in source:
    library_path.write_text(include + source)