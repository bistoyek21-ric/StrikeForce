# Install guide — curl, 7zip (terminal), SFML, libtorch
## Supported OS: Windows, Debian/Ubuntu (Linux), macOS — direct downloads + package installs included.


## 1) curl (HTTP client)


### Windows:
 • curl ships with modern Windows, but official builds & portable zips are on the curl site. If you need the latest or a specific build, download from the official Windows page and add to PATH if you used a zip.  
PowerShell check
```shell
 curl --version
```
If missing / manual install
 1. Download a Win64 zip from https://curl.se/windows/ (pick the generic/mingw build).  
 2. Extract to C:\Tools\curl\, add C:\Tools\curl\bin to System PATH (System → Environment Variables).
 3. Re-open terminal and run curl --version.


### Debian / Ubuntu:
```shell
 sudo apt update
 sudo apt install curl -y
 curl --version
```


### macOS:
 • macOS already ships with curl. Verify:
```shell
 curl --version
```
• If you want a newer build, use Homebrew:
```shell
 brew install curl
```


-------

## 2) 7zip (archive tool — CLI: 7z / p7zip)


### Windows:
 • Official installers (EXE) on 7-Zip site. For automated installs use winget or choco.  
GUI / CLI installer
```shell
 winget install --id=7zip.7zip
 #verification:
 7z
```
Or download EXE from https://www.7-zip.org/ and run the installer.  


### Debian / Ubuntu:
```shell
 sudo apt update
 sudo apt install p7zip-full p7zip-rar -y
 #verification:
 7z --help
```
(p7zip-full is the usual package for 7z CLI on Debian/Ubuntu).  


### macOS:
```shell
 brew install p7zip
 #verification:
 7z --help
```


-----------

## 3) SFML (C++ multimedia library)

Precompiled SDKs are available for Windows / Linux / macOS on the SFML site; their tutorials show how to link for Visual Studio, GCC/Clang, and CMake.  

### Windows:
 1. Download the SDK for Windows (matching your compiler/toolchain) from SFML downloads.  
 2. Extract to C:\Libraries\SFML\.
 3. In Visual Studio: add SFML\include to Include Directories, SFML\lib to Library Directories, and link sfml-graphics.lib, sfml-window.lib, sfml-system.lib (or the -s debug variants). See SFML Visual Studio guide.  
 
### Debian / Ubuntu:
```shell
sudo apt update
sudo apt install libsfml-dev -y
```
That installs headers and libs from the Debian repo. You can also download the SFML SDK from the website if you want a newer version. After install:
```shell
# quick compile test
g++ -o sfml-test sfml-test.cpp -lsfml-graphics -lsfml-window -lsfml-system
./sfml-test
```
See SFML Linux getting-started docs.  


### macOS:
```shell
brew install sfml
```
Or download macOS SDK from SFML site. For Xcode/CLion, configure include & lib paths or use the CMake template from SFML docs.  

Quick CMake hint (SFML)
```
find_package(SFML 3 COMPONENTS graphics window system REQUIRED)
target_link_libraries(myapp PRIVATE sfml::graphics sfml::window sfml::system)
```
(Use the CMake tutorial on the SFML site for version-specific notes.)  

------

## 4) libtorch (PyTorch C++ distribution)

LibTorch is provided by PyTorch as precompiled archives for different OSes and compute platforms (CPU / CUDA). You download the archive from the PyTorch site (Get Started → C++ / LibTorch). Use CMAKE_PREFIX_PATH to point to libtorch when building.  

### Windows:
 1. Go to PyTorch “Get Started” → choose C++ / LibTorch, pick CPU or CUDA build and download the .zip.  
 2. Extract to C:\Libraries\libtorch.
 3. In Visual Studio / CMake project: set include dirs to libtorch\include and libtorch\include\torch\csrc\api\include, library dir to libtorch\lib. Link the provided .lib files.
 4. Example MSBuild/CMake flag for CMake:
```shell
cmake -DCMAKE_PREFIX_PATH="C:/Libraries/libtorch" ..
```


### Debian / Ubuntu:
 1. Download the Linux libtorch .zip/.tar.gz on the PyTorch site (choose CPU or CUDA).  
 2. Extract to /opt/libtorch or ~/libtorch.
```shell
unzip libtorch-*.zip -d ~/libtorch
```
 3. CMake:
```shell
cmake -DCMAKE_PREFIX_PATH=/home/you/libtorch ..
make -j$(nproc)
```
4. See PyTorch docs for compatibility (ABIs, CUDA version).  

 
### macOS:
 1. Download macOS libtorch from PyTorch “Get Started” (note macOS builds and compatibility).  
 2. Extract and point CMAKE_PREFIX_PATH to the libtorch folder:
```shell
tar -xzf libtorch-macos-*.tar.gz
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
```
CMake snippet (libtorch)
```shell
find_package(Torch REQUIRED)
target_link_libraries(myapp "${TORCH_LIBRARIES}")
set_property(TARGET myapp PROPERTY CXX_STANDARD 17)
```
And run:
```shell
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j
```
(Use the libtorch tutorial on PyTorch site for ABI/compiler specifics.)  

------

## Verification checklist (do these after install)
 • curl --version (Windows/Unix/macOS).  
 • 7z or 7z --help (CLI) or open 7-Zip GUI on Windows.  
 • SFML: compile a minimal window example (or run ldd / otool -L to check linked libs). See SFML tutorials.  
 • libtorch: compile a trivial C++ program that #include <torch/torch.h> and prints tensor ops; if it builds & runs, you’re golden.  

-----

### Extra tips / common gotchas (pro tips)
 • Match compiler toolchain with precompiled binaries. (MSVC vs MinGW on Windows; GCC/Clang ABI on Linux/macOS). SFML & libtorch provide different builds for different toolchains—pick the right one.  
 • For libtorch GPU builds: CUDA version must match your system drivers and the libtorch CUDA build (very common source of runtime errors).  
 • Use Homebrew on macOS to keep CLI tools tidy; use apt on Debian/Ubuntu for stable packages.
 • Keep libraries in dedicated folders (C:\Libraries, /opt/lib, ~/libs) rather than scattering across system folders.
