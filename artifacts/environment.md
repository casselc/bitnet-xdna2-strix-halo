# Environment

Captured: 2026-08-30T00:32:41-07:00 on `marvin`

## Hardware
```
CPU:      AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
Cores:    32 threads, 16 cores
RAM:      122Gi
c6:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] Strix Halo [Radeon Graphics / Radeon 8050S Graphics / Radeon 8060S Graphics] [1002:1586] (rev c1)
c7:00.1 Signal processing controller [1180]: Advanced Micro Devices, Inc. [AMD] Strix/Krackan/Strix Halo Neural Processing Unit [1022:17f0] (rev 11)
```

## OS / kernel
```
Distro:   Ubuntu 26.04 LTS
Kernel:   7.0.0-29-generic
glibc:    2.43
cmdline:  BOOT_IMAGE=/BOOT/ubuntu_jk2x0w@/vmlinuz-7.0.0-29-generic root=ZFS=rpool/ROOT/ubuntu_jk2x0w ro quiet splash crashkernel=2G-4G:320M,4G-32G:512M,32G-64G:1024M,64G-128G:2048M,128G-:4096M
```

## NPU driver / firmware
```
module:   /lib/modules/7.0.0-29-generic/kernel/drivers/accel/amdxdna/amdxdna.ko.zst (intree=Y)
device:   crw-rw-rw- 1 root render 261, 0 Aug 30 00:23 /dev/accel/accel0
fw:       1.1.2.65
revision: 0x11
iommu:    ivhd0 
memlock:  unlimited
```

## XRT
```
libxrt-alveo2        1:2.25.0-4~resolute1
libxrt-dev           1:2.25.0-4~resolute1
libxrt-npu2          1:2.25.0-4~resolute1
libxrt-utils         1:2.25.0-4~resolute1
libxrt-utils-npu     1:2.25.0-4~resolute1
libxrt2              1:2.25.0-4~resolute1
python3-xrt          1:2.25.0-4~resolute1

XRT
  Version              : 2.25.00
  virtio-pci Version   : 7.0.0-29-generic
  amdxdna Version      : 7.0.0-29-generic
  NPU Firmware Version : 1.1.2.65

Device(s) Present
|BDF             |Name          |Architecture  |Topology  |
|----------------|--------------|--------------|----------|
|[0000:c7:00.1]  |RyzenAI-npu5  |aie2p         |6x8       |



Platform
  Name                   : RyzenAI-npu5 
  Power Mode             : default 
  Total Columns          : 8 
```

## Toolchain
```
gcc        gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0
clang      Ubuntu clang version 21.1.8 (6ubuntu1)
clang++    Ubuntu clang version 21.1.8 (6ubuntu1)
cmake      cmake version 4.2.3
ninja      1.13.2
python3    Python 3.14.4
pip        26.2.1
```
