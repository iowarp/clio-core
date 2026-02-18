"""CLI entry point for the chimaera command.

Locates the chimaera binary bundled in the wheel and exec's it,
ensuring IOWarp shared libraries are on the library search path.
"""

import os
import sys


def main():
    """Execute the chimaera binary."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(package_dir, "bin", "chimaera")
    lib_dir = os.path.join(package_dir, "lib")

    if not os.path.exists(bin_path):
        print("Error: chimaera binary not found at", bin_path, file=sys.stderr)
        sys.exit(1)

    # Ensure IOWarp libs are on the library path
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = (
            lib_dir + ":" + ld_path if ld_path else lib_dir
        )

    # Replace this process with the chimaera binary
    os.execve(bin_path, [bin_path] + sys.argv[1:], os.environ)


if __name__ == "__main__":
    main()
