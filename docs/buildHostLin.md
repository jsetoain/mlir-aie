# Linux Setup and Build Instructions

These instructions will guide you through everything required for building and executing a program on the Ryzen™ AI NPU, starting from a fresh bare-bones **Ubuntu 22.04 LTS** install. Only Ubuntu 22.04 LTS is supported. 

## Initial Setup

#### Update BIOS:

Be sure you have the latest BIOS for your laptop or mini PC, this will ensure the NPU (sometimes referred to as IPU) is enabled in the system. You may need to manually enable the NPU:
:
   ```Advanced → CPU Configuration → IPU``` 

> **NOTE:** Some manufacturers only provide Windows executables to update the BIOS, please do this before installing Ubuntu. 

#### BIOS Settings:
1. Turn off SecureBoot (Allows for unsigned drivers to be installed)

   ```BIOS → Security → Secure boot → Disable```

1. Turn Ac Power Loss to "Always On" (Can be used for PDU reset, turns computer back on after power loss)

   ```BIOS → Advanced → AMD CBS →  FCH Common Options → Ac Power Loss Options → Set Ac Power Loss to "Always On"```

## Overview
You will...

1. Install a driver for the Ryzen™ AI. As part of this, you will need to...

   1. [...install Xilinx Vitis and obtain a license.](#install-xilinx-vitis-20232)

   1. [...compile and install a more recent Linux kernel.](#update-linux)

   1. [...compile and install the XDNA driver from source.](#install-the-xdna-driver)

1. Install the compiler toolchain, allowing you to compile your own NPU designs from source. As part of this, you will need to...


   1. [...install prerequisites.](#install-mlir-aie-prerequisites)
   
   1. ...install MLIR-AIE [from precompiled binaries (fast)](#option-a---quick-setup-for-ryzen-ai-application-development) or [from source (slow)](#option-b---build-mlir-aie-tools-from-source-for-development).

1. Build and execute one of the example designs. This consists of...

   1. [...setting up your environment.](#setting-up-your-environment)
   
   2. [...building device (NPU) code.](#build-device-aie-part)
   
   3. [...building and executing host (x86) code and device (NPU) code.](#build-and-run-host-part) 

> Be advised that two of the steps (Linux compilation and Vitis install) may take hours. If you decide to build mlir-aie from source, this will also take a long time as it contains an LLVM build. Allocate enough time and patience. Once done, you will have an amazing toolchain allowing you to harness this great hardware at your hands.

## Prerequisites

### Install Xilinx Vitis 2023.2 

1. Install Vitis under from [Xilinx Downloads](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html). You will need to run the installer as root. We will assume you use the default installation directory, `/tools/Xilinx`.

   > This is a large download. A wired connection will speed things up. Be prepared to spend multiple hours on this step.

1. Set up a AI Engine license.

    1. Get a local license for AIE Engine tools from [https://www.xilinx.com/getlicense](https://www.xilinx.com/getlicense).

    1. Copy your license file (Xilinx.lic) to your preferred location, e.g. `/opt/Xilinx.lic`:
       
    1. Setup your environment using the following script for Vitis and aietools:

       ```bash
       #!/bin/bash
        #################################################################################
        # Setup Vitis (which includes Vitis and aietools)
        #################################################################################
        export MYXILINX_VER=2023.2
        export MYXILINX_BASE=/tools/Xilinx
        export XILINX_LOC=$MYXILINX_BASE/Vitis/$MYXILINX_VER
        export AIETOOLS_ROOT=$XILINX_LOC/aietools
        export PATH=$PATH:${AIETOOLS_ROOT}/bin:$XILINX_LOC/bin
        export LM_LICENSE_FILE=/opt/Xilinx.lic
        export VITIS=${XILINX_LOC}
        export XILINX_VITIS=${XILINX_LOC}
        export VITIS_ROOT=${XILINX_LOC}
       ```
   1. Vitis requires some python3.8 libraries:
  
      ```bash
      sudo add-apt-repository ppa:deadsnakes/ppa
      sudo apt-get update
      sudo apt install libpython3.8-dev
      ```

### Update Linux

> The reason we need to update the kernel is that the XDNA driver requires IOMMU SVA support.

1. Disable **Secure Boot** in the BIOS. This allows for unsigned drivers to be installed.

    >  On the ASUS Vivobook, this setting can be found under
      BIOS → Advanced Settings (F7) → Security →  Secure Boot → Secure Boot Control (Set to Disabled)

1. Install the following prerequisite packages for compiling Linux:
    ```bash
    sudo apt install \
    build-essential debhelper flex bison libssl-dev libelf-dev libboost-all-dev libpython3.10-dev libsystemd-dev libtiff-dev libudev-dev
    ```

1. Pull the source for kernel version 6.10.

    ```bash
    git clone --depth=1 --branch v6.10 git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
    export LINUX_SRC_DIR=$(realpath linux)
    ```

1. Create a build directory and a configuration within it.
    
    ```bash
    mkdir linux-build
    export LINUX_BUILD_DIR=$(realpath linux-build)
    cp /boot/config-`uname -r` $LINUX_BUILD_DIR/.config
    ```

1. Go to the directory where you cloned Linux and adjust the configuration.

    ```bash
    cd $LINUX_SRC_DIR
    ./scripts/config --file $LINUX_BUILD_DIR/.config --disable MODULE_SIG
    ./scripts/config --file $LINUX_BUILD_DIR/.config --disable SYSTEM_TRUSTED_KEYS
    ./scripts/config --file $LINUX_BUILD_DIR/.config --disable SYSTEM_REVOCATION_KEYS
    ./scripts/config --file $LINUX_BUILD_DIR/.config --enable DRM_ACCEL
    make O=$LINUX_BUILD_DIR olddefconfig
    ```

1. Build Linux.

    ```bash
    make -j$(nproc) O=$LINUX_BUILD_DIR bindeb-pkg 2>&1 | tee kernel-build.log
    ```

    > Compiling the linux kernel may take hours.
    
    > Note that the final kernel `.deb` packages will be in the *parent* directory of `LINUX_BUILD_DIR`.

1. Install the new Linux kernel and reboot.

    ```bash
    cd $LINUX_BUILD_DIR
    sudo apt reinstall ../linux-headers-6.10.0_6.10.0-1_amd64.deb ../linux-image-6.10.0_6.10.0-1_amd64.deb ../linux-libc-dev_6.10.0-1_amd64.deb
    sudo shutdown --reboot 0
    ```

### Install the XDNA Driver

1. Install a more recent CMake, which is needed for building XRT.
   
   1. Download CMake 3.28 binaries into `NEW_CMAKE_DIR`.
      ```bash
      mkdir cmake
      export NEW_CMAKE_DIR=$(realpath cmake)
      cd cmake
      wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.sh
      chmod +x ./cmake-3.28.3-linux-x86_64.sh
      ./cmake-3.28.3-linux-x86_64.sh
      ```

   1. Answer the prompts with **y** (accept license), then **n** (include subdirectory).

   1. Add new cmake directory to your `PATH`.

      ```bash
      export PATH="${NEW_CMAKE_DIR}/bin":"${PATH}"
      ```
   
   1. Verify the install of CMake was successful.

      ```bash
      cmake --version
      ```

      > The frist line this prints should read
      > ```cmake version 3.28.3```

1. Install the following prerequisite packages.
 
   ```bash
   sudo apt install \
   libidn11-dev
   ```

1. Clone the XDNA driver repository and its submodules.
    ```bash
    git clone https://github.com/amd/xdna-driver.git
    export XDNA_SRC_DIR=$(realpath xdna-driver)
    cd xdna-driver
    git reset --hard 537a509a3ab1b698c9c9f6ebcd88035b2fe8359b
    git submodule update --init --recursive
    ```

    > The submodules use SSH remotes. You will need a GitHub account and locally installed SSH keys to pull the submodules. Follow [these instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) to set up an SSH key. Alternatively, edit `.gitmodules` to use HTTPS instead of SSH.

1. Install XRT. (Below steps are adapted from [here](https://xilinx.github.io/XRT/master/html/build.html).)

    1. Install XRT prerequisites.
    
       ```bash
       cd $XDNA_SRC_DIR
       sudo ./tools/amdxdna_deps.sh
       ```

    2. Build XRT. Remember to source the aietools/Vitis setup script from [above](#install-xilinx-vitis-20232).

       ```bash
       cd $XDNA_SRC_DIR/xrt/build
       ./build.sh -noert -noalveo
       ```

    3. Install XRT.

       ```bash
       cd $XDNA_SRC_DIR/xrt/build/Release
       sudo apt reinstall ./xrt_202420.2.18.0_22.04-amd64-xrt.deb ./xrt_202420.2.18.0_22.04-amd64-xbflash.deb
       ```

       > **An error is expected in this step.** Ignore it.



1. Build XDNA-Driver. Below steps are adapted from [here](https://github.com/amd/xdna-driver).

    ```bash
    cd $XDNA_SRC_DIR/build
    ./build.sh -release
    ./build.sh -package
    ```

1. Install XDNA.

    ```bash
    cd $XDNA_SRC_DIR/build/Release
    sudo apt reinstall ./xrt_plugin.2.18.0_ubuntu22.04-x86_64-amdxdna.deb
    ```
    
1. Check that the NPU is working if the device appears with xrt-smi:
   
   ```bash
   source /opt/xilinx/xrt/setup.sh
   xrt-smi examine
   ```

   > At the bottom of the output you should see:
   >  ```
   >  Devices present
   >  BDF             :  Name             
   > ------------------------------------
   >  [0000:66:00.1]  :  RyzenAI-npu1
   >  ```

### Install MLIR-AIE Prerequisites

1. Install the following packages needed for MLIR-AIE:

    ```bash
    sudo apt install \
    build-essential clang clang-14 lld lld-14 cmake python3-venv python3-pip libxrender1 libxtst6 libxi6 virtualenv
    ```

1. Install g++13 and opencv needed for some programming examples:

   ```bash
   sudo add-apt-repository ppa:ubuntu-toolchain-r/test
   sudo apt update
   sudo apt install gcc-13 g++-13 -y
   sudo apt install libopencv-dev python3-opencv
   ```

1. Remember to source the aietools/Vitis setup script from [above](#install-xilinx-vitis-20232).

1. Choose *one* of the two options (A or B) below for installing MLIR-AIE.

### Option A - Quick Setup for Ryzen™ AI Application Development

1. Clone [the mlir-aie repository](https://github.com/Xilinx/mlir-aie.git), best under /home/username for speed (yourPathToBuildMLIR-AIE): 
   ```bash
   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie
   ````

1. Source `utils/quick_setup.sh` to setup the prerequisites and
   install the mlir-aie and llvm compiler tools from whls.

1. Jump ahead to [Build Device AIE Part](#build-device-aie-part) step 2 below.

### Option B - Build mlir-aie Tools from Source for Development

1. Clone [https://github.com/Xilinx/mlir-aie.git](https://github.com/Xilinx/mlir-aie.git) best under /home/username for speed (yourPathToBuildMLIR-AIE), with submodules: 
   ```bash
   git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
   ````

1. Follow regular getting started instructions [Building on x86](https://xilinx.github.io/mlir-aie/Building.html) from step 2. Please disregard any instructions referencing alternative LibXAIE versions or sysroots.

## Setting up your Environment

After all prerequisites (drivers and compilation toolchain) have been installed, you need to make them findable by adding them to the `PATH` and setting required environment variables.

We suggest you add all of the following to a `setup.sh` script in your home directory, and `source setup.sh` as the first step of your workflow. That way, everything is set up in one setp.


### `setup.sh` - Option A - Quick Setup

```bash
export LM_LICENSE_FILE=/opt/Xilinx.lic
source /opt/xilinx/xrt/setup.sh
export PATH="${NEW_CMAKE_DIR}/bin":"${PATH}"

cd ${MLIR_AIE_BUILD_DIR}
source ${MLIR_AIE_BUILD_DIR}/ironenv/bin/activate
source ${MLIR_AIE_BUILD_DIR}/utils/env_setup.sh ${MLIR_AIE_BUILD_DIR}/my_install/mlir_aie ${MLIR_AIE_BUILD_DIR}/my_install/mlir
```

> Replace `${MLIR_AIE_BUILD_DIR}` with the directory in which you *built* mlir-aie above. Replace `${NEW_CMAKE_DIR}` with the directory in which you installed CMake 3.28 above. Instead of search and replace, you can also define these values as environment variables.

> For quick setup, this step is only needed if you are starting with a new terminal. If you are continuing in the same terminal you used to install the prerequisites, the environment variables should all be set.

### `setup.sh` - Option B - Toolchain Compiled From Source

```bash
cd ${MLIR_AIE_BUILD_DIR}
source ${MLIR_AIE_BUILD_DIR}/sandbox/bin/activate
source /opt/xilinx/xrt/setup.sh
source ${MLIR_AIE_BUILD_DIR}/utils/env_setup.sh ${MLIR_AIE_BUILD_DIR}/install ${MLIR_AIE_BUILD_DIR}/llvm/install
```

> Replace `${MLIR_AIE_BUILD_DIR}` with the directory in which you *built* mlir-aie above. Instead of search and replace, you can also define `MLIR_AIE_BUILD_DIR` as an environment variable.

## Build a Design

For your design of interest, for instance from [programming_examples](../programming_examples/), 2 steps are needed: (i) build the AIE desgin and then (ii) build the host code.

### Build Device AIE Part

1. Prepare your enviroment with the mlir-aie tools (built during prerequisites part of this guide) - see **"Setting Up Your Environment"** avove.

2. Goto the design of interest and run `make`

### Build and Run Host Part

Note that your design of interest might need an adapted `CMakeLists.txt` file. Also pay attention to accurately set the paths CMake parameters `BOOST_ROOT`, `XRT_INC_DIR` and `XRT_LIB_DIR` used in the `CMakeLists.txt`, either in the file or as CMake command line parameters.

1. Build: Goto the same design of interest folder where the AIE design just got built (see above)
    ```bash
    make <testName>.exe
    ```
    > Note that the host code target has a `.exe` file extension even on Linux. Although unusual, this is an easy way for us to distinguish whether we want to compile device code or host code.


1. Run (program arguments are just an example for add_one design)
    ```bash
    cd Release
    .\<testName>.exe -x ..\..\build\final.xclbin -k MLIR_AIE -i ..\..\build\insts.txt -v 1
    ```

# Troubleshooting

## Signing your XCLBIN (older xdna Linux drivers)

1. Signing your array configuration binary aka. XCLBIN
    ```bash
    sudo bash
    source /opt/xilinx/xrt/setup.sh
    # Assume adding an unsigned xclbin on Phoenix, run
    /opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh -dev Phoenix -xclbin <your test>.xclbin

    # <your test>_unsigned.xclbin will be added into /lib/firmware/amdxdna/<version>/ and symbolic link will create.
    # When xrt_plugin package is removed, it will automatically cleanup.
    ```
    1. Alternatively, you can `sudo chown -R $USER /lib/firmware/amdnpu/1502/` and remove the check for root in `/opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh` (look for `!!! Please run as root !!!`).

## Resetting the NPU

It is possible to hang the NPU in an unstable state. To reset the NPU:

```bash
sudo rmmod amdxdna.ko
sudo insmod $XDNA_SRC_DIR/build/Release/bins/driver/amdxdna.ko
```

If you installed the AMD XDNA driver using `.deb` packages as outlined above, and `insmod` does not work, you may instead want to try:

```bash
sudo modprobe -r amdxdna
sudo modprobe -v amdxdna
```

## `xrt_core::system_error` - Unsigned xclbins

If you are able to successfully build your design, but are getting the following error when trying to execute it:

```
terminate called after throwing an instance of 'xrt_core::system_error'
  what():  DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=2): No such file or directory
Aborted (core dumped)
```

This may be because you did not sign your `final.xclbin`. The device only allows executing signed xclbins for some versions of the driver. Follow step 3 under section [Build Device AIE Part](#build-device-aie-part) above.

## Signing the `xclbin` hangs

As outlined above, `.xclbin` files must be signed to be able to run on the device. Signing is done by running

```
/opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh -dev Phoenix -xclbin <your test>.xclbin
```

This may hang after the following output if you have too many signed `.xclbin`s:

```
Copy <your test>.xclbin to /lib/firmware/amdnpu/1502/<your test>.xclbin
```

If this happens, clear all your previously signed `.xclbin`s as follows (you will of course have to re-sign the ones you remove in this step if you want to run them again, but chances are you have many old unneeded `.xclbin`s in there):

```
rm /lib/firmware/amdnpu/1502/<your tests>.xclbin
```

## License Errors When Trying to Compile

The `v++` compiler for the NPU device code requires a valid Vitis license. If you are getting errors related to this:

1. You have obtained a valid license, as described [above](#install-xilinx-vitis-20232-and-other-mlir-aie-prerequisites). 
1. Make sure you have set the environment variable `LM_LICENSE_FILE` to point to your license file, see [above](#setting-up-your-environment).
1. Make sure the ethernet interface whose MAC address you used to generate the license is still available on your machine. For example, if you used the MAC address of a removable USB Ethernet adapter, and then removed that adapter, the license check will fail. You can list MAC addresses of interfaces on your machine using `ip link`.

-----

<p align="center">Copyright&copy; 2019-2024 AMD</p>
