# StrikeForce — Installation Guide

This guide takes you from a clean computer to a running StrikeForce client. You will install a few tools and libraries, tell your system where they are, then build and run the game with **one single command** — no CMake, no Makefile, no IDE:

```bash
cd StrikeForce-client
g++ -std=c++17 main.cpp -o app -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lsfml-graphics -lsfml-window -lsfml-system && ./app
```

(`app` is just the name of the executable — you can call it anything.)

Because that command has **no** `-I` (include) or `-L` (library) flags, the compiler must find everything through **environment variables**. Most of this guide is about setting those up correctly. Once that is done, the command above just works.

---

## Table of contents

1. [What you are installing](#1-what-you-are-installing)
2. [Which OS path should I follow?](#2-which-os-path-should-i-follow)
3. [Linux (Debian / Ubuntu)](#3-linux-debian--ubuntu)
4. [macOS](#4-macos)
5. [Windows (via WSL2)](#5-windows-via-wsl2)
6. [Build and run StrikeForce](#6-build-and-run-strikeforce)
7. [Troubleshooting](#7-troubleshooting)
8. [Optional: verify each library separately](#8-optional-verify-each-library-separately)

---

## 1. What you are installing

| Tool / library | Why StrikeForce needs it |
|---|---|
| **g++** (C++17 compiler) | Compiles the game. |
| **curl** | The game calls it to download/upload AI checkpoints (backup subsystem). |
| **7-Zip** (`7z` command) | The game calls it to unpack/pack checkpoint archives. |
| **SFML 3.x** | Window, graphics and keyboard input. |
| **LibTorch 2.0.0** | PyTorch C++ library; runs the AI agents. |

> **Recommended LibTorch version:** `libtorch-cxx11-abi-shared-with-deps-2.0.0+cu117`
> (This is the version the project is developed with. It works on machines **without** an NVIDIA GPU too — it simply runs on the CPU.)

> **SFML 3 is required.** `GraphicPrinter.hpp` uses the SFML 3 API (`sf::VideoMode({W, H})`, `pollEvent()` returning `std::optional`, `sf::Text(font, ...)`, `openFromFile`, ...). SFML 2.x — which is what `apt install libsfml-dev` gives you on Ubuntu 22.04 / 24.04 — will **not** compile the project. On Linux you therefore build SFML 3 once from source (Section 3.1b); on macOS Homebrew's `sfml` package is already version 3.

---

## 2. Which OS path should I follow?

| Your system | What to do |
|---|---|
| **Ubuntu / Debian Linux** | Follow [Section 3](#3-linux-debian--ubuntu). This is the primary, best-tested path. |
| **macOS** | Follow [Section 4](#4-macos). Needs one small extra trick (empty "stub" libraries) because macOS LibTorch has no CUDA files. |
| **Windows** | Follow [Section 5](#5-windows-via-wsl2): install WSL2 (Ubuntu inside Windows), then follow the Linux steps. |

> **Why WSL2 on Windows?** Official LibTorch for Windows is built with Microsoft's Visual C++ compiler (MSVC). `g++` produces code that is binary-incompatible with it, so the exact `g++` command cannot link against it natively. WSL2 gives you a real Ubuntu where the command works unchanged, and it runs the SFML window on Windows 11 (and on up-to-date Windows 10) out of the box. If you prefer Visual Studio, you will need a different build setup — that is outside the scope of this guide.

---

## 3. Linux (Debian / Ubuntu)

Every command below goes in a normal terminal. Lines starting with `#` are comments.

### 3.1 Compiler, curl, 7-Zip, build tools

```bash
sudo apt update
sudo apt install -y g++ curl p7zip-full p7zip-rar unzip wget git cmake \
    libx11-dev libxrandr-dev libxcursor-dev libxi-dev libudev-dev \
    libfreetype-dev libgl1-mesa-dev libopenal-dev libflac-dev libvorbis-dev
```

Check that everything is installed:

```bash
g++ --version        # should print a version (9 or newer; 11 or 12 is ideal)
curl --version
7z                   # prints "7-Zip ..." usage text
cmake --version
git --version
```

### 3.1b Build and install SFML 3 (one time)

`cmake` is used **only here**, once, to build the SFML library. StrikeForce itself is still built with the single `g++` command.

```bash
mkdir -p ~/libs
cd ~/libs
git clone --depth 1 --branch 3.0.0 https://github.com/SFML/SFML.git sfml3-src
cd sfml3-src
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_PREFIX="$HOME/libs/sfml3"
cmake --build build -j"$(nproc)"
cmake --install build
```

(Any newer `3.0.x` tag is fine too.) Check the result:

```bash
ls ~/libs/sfml3/include/SFML        # Graphics.hpp  Window.hpp ...
ls ~/libs/sfml3/lib | grep sfml     # libsfml-graphics.so ...   (on some systems the folder is lib64)
```

> If your distribution already ships SFML 3 as a package (`apt show libsfml-dev` shows version 3.x), you can install that instead and skip this subsection — then you also do not need the `SFML3` lines in the next step.

### 3.2 Download and extract LibTorch

We will keep all libraries in one folder, `~/libs`.

```bash
mkdir -p ~/libs
cd ~/libs

# Recommended build (CUDA 11.7 build; also runs fine on CPU-only machines):
wget "https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu117.zip" \
     -O libtorch.zip

unzip -q libtorch.zip
rm libtorch.zip
```

After this you should have `~/libs/libtorch/` containing the folders `include`, `lib`, `share`.

```bash
ls ~/libs/libtorch
# build-hash  build-version  include  lib  share
ls ~/libs/libtorch/lib | grep -E "libtorch_cuda|libc10_cuda"
# libc10_cuda.so
# libtorch_cuda.so
```

> **No NVIDIA GPU / want a smaller download?** You can use the CPU-only build instead:
> `https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip`
> But then `libtorch_cuda.so` and `libc10_cuda.so` do not exist, and the build command (which asks for them) would fail. If you choose the CPU-only build, also do the **stub libraries** step in [3.3](#33-only-if-you-use-the-cpu-only-libtorch-create-stub-libraries). With the recommended `cu117` build you can skip it.

> **NVIDIA GPU users:** you only need a working NVIDIA driver (version 515 or newer is recommended for CUDA 11.7). You do **not** need to install the CUDA Toolkit — the CUDA runtime libraries are already inside the `with-deps` LibTorch zip.

### 3.3 (Only if you use the CPU-only LibTorch) create stub libraries

Skip this subsection if you downloaded the `cu117` build.

```bash
mkdir -p ~/libs/stubs
cd ~/libs/stubs
echo "" | g++ -shared -fPIC -x c++ - -o libtorch_cuda.so
echo "" | g++ -shared -fPIC -x c++ - -o libc10_cuda.so
```

These are empty libraries. They only exist so the linker does not complain about `-ltorch_cuda -lc10_cuda`.
Then add `$HOME/libs/stubs` to the paths in the next step (the block below already has it, commented out — remove the `#` in front of the stub lines).

### 3.4 Tell your system where everything is (environment variables)

Open your shell config file in an editor:

```bash
nano ~/.bashrc          # if you use bash (the default on Ubuntu)
# nano ~/.zshrc         # if you use zsh
```

Add this block at the very bottom (copy it exactly):

```bash
# ============== StrikeForce: LibTorch + SFML 3 paths ==============
export LIBTORCH="$HOME/libs/libtorch"
export SFML3="$HOME/libs/sfml3"

# 1) Where the compiler finds the headers (#include <torch/torch.h>)
export CPLUS_INCLUDE_PATH="$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include:$SFML3/include:$CPLUS_INCLUDE_PATH"

# 2) Where the linker finds the libraries at build time (-ltorch ...)
export LIBRARY_PATH="$LIBTORCH/lib:$SFML3/lib:$LIBRARY_PATH"

# 3) Where the program finds the libraries when it runs
export LD_LIBRARY_PATH="$LIBTORCH/lib:$SFML3/lib:$LD_LIBRARY_PATH"

# --- Only for the CPU-only LibTorch (remove the leading '#' on these 2 lines) ---
# export LIBRARY_PATH="$HOME/libs/stubs:$LIBRARY_PATH"
# export LD_LIBRARY_PATH="$HOME/libs/stubs:$LD_LIBRARY_PATH"
# =========================================================
```

Save (in nano: `Ctrl+O`, `Enter`, `Ctrl+X`), then apply it to the current terminal:

```bash
source ~/.bashrc        # or: source ~/.zshrc
```

Check it worked:

```bash
echo $LIBTORCH
# /home/<your-name>/libs/libtorch
```

> If your SFML 3 install put its libraries in `lib64` instead of `lib`, use `$SFML3/lib64` in the two library lines above.

Now go to [Section 6 — Build and run](#6-build-and-run-strikeforce).

---

## 4. macOS

Open **Terminal**. You need Homebrew; if you do not have it, install it from <https://brew.sh> first.

### 4.1 Compiler, curl, 7-Zip, SFML

```bash
xcode-select --install          # installs the compiler (Apple clang). Skip if already installed.
brew install curl p7zip wget
brew install sfml               # must be version 3.x (check: brew info sfml)
brew install libomp             # OpenMP runtime used by LibTorch
```

On macOS, the command `g++` is actually Apple's clang with the same options — this is normal and works fine with the StrikeForce command.

Check:

```bash
g++ --version
curl --version
7z                              # prints "7-Zip ..." usage text
brew info sfml | head -1        # should say 3.x
```

### 4.2 Download and extract LibTorch

Go to <https://pytorch.org/get-started/locally/>, choose **Stable → LibTorch → C++/Java → Mac → CPU**, and download the zip. Pick the build that matches your Mac:

- **Apple Silicon (M1/M2/M3/M4):** use the **arm64** LibTorch build.
- **Intel Mac:** use the standard x86_64 build.

> The project's recommended version is **2.0.0**. Older versions may not offer an arm64 build for macOS; if so, use the **oldest 2.x version that has an arm64 download** (2.0.1 or newer). Mixed minor versions are normally fine for compiling StrikeForce.

Extract it into `~/libs`:

```bash
mkdir -p ~/libs
cd ~/libs
unzip -q ~/Downloads/libtorch-macos-*.zip      # adjust the file name if different
ls ~/libs/libtorch                              # should show: include  lib  share ...
```

### 4.3 Create stub libraries (required on macOS)

macOS LibTorch has no CUDA, but the build command asks for `-ltorch_cuda` and `-lc10_cuda`. We create two empty libraries so the linker is satisfied:

```bash
mkdir -p ~/libs/stubs
cd ~/libs/stubs
echo "" | g++ -shared -x c++ - -o libtorch_cuda.dylib
echo "" | g++ -shared -x c++ - -o libc10_cuda.dylib
ls
# libc10_cuda.dylib  libtorch_cuda.dylib
```

### 4.4 Tell your system where everything is

macOS uses **zsh** by default:

```bash
nano ~/.zshrc
```

Add this block at the bottom:

```bash
# ============== StrikeForce: LibTorch + SFML paths ==============
export LIBTORCH="$HOME/libs/libtorch"
export SFML="$(brew --prefix sfml)"

# 1) Headers
export CPLUS_INCLUDE_PATH="$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include:$SFML/include:$CPLUS_INCLUDE_PATH"

# 2) Libraries at build time
export LIBRARY_PATH="$LIBTORCH/lib:$SFML/lib:$HOME/libs/stubs:$(brew --prefix libomp)/lib:$LIBRARY_PATH"

# 3) Libraries at run time
export DYLD_FALLBACK_LIBRARY_PATH="$LIBTORCH/lib:$SFML/lib:$HOME/libs/stubs:$(brew --prefix libomp)/lib:$DYLD_FALLBACK_LIBRARY_PATH"
# ================================================================
```

Save and apply:

```bash
source ~/.zshrc
echo $LIBTORCH
```

> macOS may block the downloaded LibTorch files the first time ("cannot be opened because the developer cannot be verified"). Fix it once with:
> `xattr -dr com.apple.quarantine ~/libs/libtorch`

Now go to [Section 6 — Build and run](#6-build-and-run-strikeforce).

---

## 5. Windows (via WSL2)

You will run StrikeForce inside Ubuntu on Windows. It takes about 10 minutes.

### 5.1 Install WSL2 with Ubuntu

1. Open **PowerShell as Administrator** (right-click Start → "Terminal (Admin)").
2. Run:
   ```powershell
   wsl --install -d Ubuntu
   ```
3. Restart the computer if asked. After restart, an Ubuntu window opens — choose a username and password.

> Requirements: Windows 11, or Windows 10 version 22H2 with the latest updates. Windows 11 / recent Windows 10 builds include **WSLg**, which lets Linux windows (like StrikeForce's SFML window) appear on your Windows desktop automatically.

### 5.2 Use the Linux instructions inside Ubuntu

Open the **Ubuntu** app from the Start menu. Everything from here happens in that Ubuntu terminal. Follow **[Section 3 (Linux)](#3-linux-debian--ubuntu) completely** — sections 3.1, 3.2, 3.4 (and 3.3 if you chose CPU-only), then [Section 6](#6-build-and-run-strikeforce).

Extra Windows notes:

- **Where to put the project:** clone/copy StrikeForce **inside the Linux file system** (e.g. `~/StrikeForce`), not under `/mnt/c/...`. It is much faster and avoids file-permission issues.
  ```bash
  cd ~
  git clone https://github.com/bistoyek21-ric/StrikeForce.git
  ```
  (If git is missing: `sudo apt install -y git`.)
- **NVIDIA GPU:** install the normal NVIDIA driver on **Windows** (not inside Ubuntu). WSL2 uses it automatically. The `cu117` LibTorch already contains the CUDA runtime.
- **Opening the project folder in Windows Explorer:** type `\\wsl$\Ubuntu\home\<your-name>\StrikeForce` in the Explorer address bar.

### 5.3 Native Windows (not supported by this guide)

The exact `g++` command cannot link the official Windows LibTorch (MSVC-built) — see the note in [Section 2](#2-which-os-path-should-i-follow). Use WSL2.

---

## 6. Build and run StrikeForce

Open a **new** terminal (or run `source ~/.bashrc` / `source ~/.zshrc`) so the environment variables from the previous steps are active. Then:

```bash
cd StrikeForce/StrikeForce-client        # adjust to where you cloned the repo

g++ -std=c++17 main.cpp -o app -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lsfml-graphics -lsfml-window -lsfml-system

./app
```

Or combine both in one line, as in the project README:

```bash
g++ -std=c++17 main.cpp -o app -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lsfml-graphics -lsfml-window -lsfml-system && ./app
```

Notes:

- **The first compile is slow** (often 1–5 minutes and lots of RAM) because LibTorch headers are huge. This is normal.
- Run `./app` **from inside `StrikeForce-client`** — the game reads folders such as `./map`, `./Items` and `./accounts`, and the font `./DejaVuSansMono.ttf`, relative to the current directory.
- **The graphics window does not open at program start.** The menus are in the terminal; the SFML window opens when a match begins (after you pick a game mode) and closes when it ends.
- A quick way to try the game: sign in with username `1` and password `1` (a ready-made test account), then pick **Solo**.
- To use a different executable name, replace `app` after `-o` (and in `./app`). Nothing else changes.

---

## 7. Troubleshooting

**`fatal error: torch/torch.h: No such file or directory`**
The include path is not set. Run `echo $CPLUS_INCLUDE_PATH` — it must contain `.../libtorch/include`. Make sure you ran `source ~/.bashrc` (or `~/.zshrc`) or opened a new terminal.

**`fatal error: SFML/Graphics.hpp: No such file or directory`**
- macOS: `brew install sfml` and make sure `$SFML/include` is in `CPLUS_INCLUDE_PATH` (Section 4.4).
- Linux/WSL: SFML 3 is not installed or not on the path — redo Section 3.1b and check `echo $CPLUS_INCLUDE_PATH` contains `.../sfml3/include`.
- Errors about `sf::VideoMode`, `pollEvent` or `sf::Text` constructors mean the compiler found **SFML 2** (e.g. the system `libsfml-dev`). Remove it (`sudo apt remove libsfml-dev`) or make sure the SFML 3 include path comes first.

**`cannot find -ltorch_cuda` / `cannot find -lc10_cuda`**
You are using a CPU-only or macOS LibTorch. Create the stub libraries (Section 3.3 or 4.3) and make sure the `stubs` folder is in `LIBRARY_PATH`.

**`cannot find -ltorch`, `-ltorch_cpu`, `-lc10`**
`LIBRARY_PATH` is not set or points to the wrong folder. `ls $LIBTORCH/lib` should list `libtorch.so` (Linux) or `libtorch.dylib` (macOS).

**`cannot find -lsfml-graphics` (macOS)**
`$SFML/lib` is missing from `LIBRARY_PATH`. Re-check Section 4.4.

**The program compiles but fails when run: `error while loading shared libraries: libtorch_cpu.so`** (Linux) or `Library not loaded: @rpath/libtorch_cpu.dylib` (macOS)
The run-time path is missing. Check `echo $LD_LIBRARY_PATH` (Linux/WSL) or `echo $DYLD_FALLBACK_LIBRARY_PATH` (macOS) — it must contain the LibTorch `lib` folder.

**Weird compile errors inside `c10/...` such as `'uint8_t' does not name a type`** (usually with GCC 13 or newer)
LibTorch 2.0.0 is older than your compiler. Install an older GCC and make `g++` point to it:
```bash
sudo apt install -y g++-11
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
sudo update-alternatives --config g++      # choose g++-11
```

**`undefined reference to pthread_...` (older Linux)**
Install a newer system, or add `-pthread` to the command. Modern Ubuntu (22.04+) does not need it.

**`Segmentation fault` or `CUDA error` at start on a machine with an old NVIDIA driver**
Update the NVIDIA driver, or use the CPU-only LibTorch (Section 3.2 note + Section 3.3).

**The match starts but no graphics window appears (WSL2)**
Remember the window only opens after you choose a game mode. If it still does not appear, go through these checks in order:
1. In PowerShell: `wsl -l -v` — the Ubuntu line must say **VERSION 2** (WSL1 has no graphics). Then `wsl --update` and `wsl --shutdown`, and reopen Ubuntu. You need Windows 11 or an up-to-date Windows 10 (WSLg).
2. In Ubuntu: `echo $DISPLAY` should print `:0` (and `ls /mnt/wslg` should exist). If it is empty or points elsewhere (e.g. left over from an old VcXsrv/X410 setup), `unset DISPLAY` or fix your `~/.bashrc`.
3. `sudo apt install -y x11-apps mesa-utils`, then run `xeyes` and `glxgears`. If these do not show windows, the problem is WSL/WSLg itself, not StrikeForce.
4. If they work but StrikeForce still shows nothing, try software OpenGL: `LIBGL_ALWAYS_SOFTWARE=1 ./app` (and `sudo apt install -y libgl1-mesa-dri libglx-mesa0`).
5. Read the terminal output: SFML prints errors such as `Failed to open X11 display` or `Failed to create an OpenGL context` there.
6. Check `macros.hpp`: if `HIGHLY_OPTIMIZED` is defined, rendering is disabled on purpose.
7. If the program asks you to "press enter" right at launch, the font `DejaVuSansMono.ttf` was not found in the current folder.

**`7z: command not found` or `curl: command not found` at run time**
Install them as shown in Sections 3.1 / 4.1. The game calls these two programs for checkpoint download/upload.

---

## 8. Optional: verify each library separately

If something fails, test each piece on its own using the **same environment variables**.

**LibTorch test** — save as `test_torch.cpp`:

```cpp
#include <torch/torch.h>
#include <iostream>
int main() {
    std::cout << torch::rand({2, 3}) << std::endl;
}
```

```bash
g++ -std=c++17 test_torch.cpp -o test_torch -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda && ./test_torch
```

You should see a 2×3 matrix of random numbers.

**SFML test** — save as `test_sfml.cpp`:

```cpp
#include <SFML/Graphics.hpp>
int main() {
    sf::RenderWindow window(sf::VideoMode({300, 200}), "SFML works!");
    while (window.isOpen()) {
        while (const std::optional e = window.pollEvent())
            if (e->is<sf::Event::Closed>()) window.close();
        window.clear(sf::Color::Green);
        window.display();
    }
}
```

```bash
g++ -std=c++17 test_sfml.cpp -o test_sfml -lsfml-graphics -lsfml-window -lsfml-system && ./test_sfml
```

A small green window should appear. Close it to exit.

**Tools test:**

```bash
curl --version
7z
```

If all four tests pass, the StrikeForce build command in [Section 6](#6-build-and-run-strikeforce) will work.