# Linux 安装指引（Vol3DGS 对比基线环境）

> 面向"另一台 Linux 机器"复现本文的对比基线环境。配套 `requirements.txt`（已适配 torch 2.11 / Blackwell）。
> 结论:Windows 上折腾的 MSVC/vcvars/`DISTUTILS_USE_SDK`/`ProgramFiles(x86)` 那套**在 Linux 上全部不需要**;
> 但下面第 1–4 条是 Linux 特有的新风险点,装前请逐条核对。

## 0. 三步安装（与 Windows 相同）

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                       # 国内建议加 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-build-isolation \
    ./submodules/diff-gaussian-rasterization ./submodules/simple-knn ./submodules/fused-ssim
pip install -e ./submodules/slang-gaussian-rasterization
```

Linux 上 `--no-build-isolation` **仍然需要**（三个扩展的 setup.py 会 `import torch`，pip 的隔离构建环境里没有 torch）；
但不再需要任何 MSVC 环境变量——带 nvcc 的工具链即可。

## 1. ⚠️ Python 必须是 3.12

`requirements.txt` 钉 `numpy==1.26.4`（上游要求 numpy<2），而 **numpy 1.26.4 没有 Python 3.13 的 wheel**。
用 3.13 会一路编译失败。`requires_python` 层面 slangtorch 允许 ≥3.10，但 numpy<2 这条把上限锁在 3.12。

## 2. ⚠️ CUDA Toolkit 12.8 + 驱动 ≥ 570

- `requirements.txt` 用 cu128 的 torch wheel，**本地编译子模块需要同版本 nvcc**（12.8），否则可能报 CUDA 版本不匹配；
- CUDA 12.8 要求 NVIDIA 驱动 **≥ 570（Linux）**；装前先 `nvidia-smi` 看驱动与 GPU，`nvcc --version` 看工具链；
- GPU 架构：cu128 wheel 覆盖 sm_75 及以上（含 Blackwell/sm_120）。若另一台机器是 **V100（sm_70）** 等更老卡，需先确认 torch 2.11 是否仍带对应内核（大概率需换旧 torch 版本，此时本案的 cu128 组合不适用）。

## 3. ⚠️ gcc 版本与 nvcc 的兼容性

CUDA 12.8 的 nvcc 对 gcc 有上限（发行版自带 gcc 14/15 时可能拒绝）。两种处理：
- 装兼容版本：`sudo apt install g++-13` 并 `export CXX=/usr/bin/g++-13 CC=/usr/bin/gcc-13`；
- 或允许不受支持的编译器：`export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"`（Windows 上我们就是这么绕过的）。

## 4. ⚠️ slangtorch 必须是 ≥1.3.22

上游钉的 1.3.7 与 torch 2.11 **不兼容**（`_write_ninja_file_and_build_library` 新增必填参数 → JIT 期 TypeError）。
1.3.22 有 Linux wheel（`manylinux_2_27_x86_64`，需 glibc ≥ 2.27，即 Ubuntu 18.04+）；它对 nvcc/g++ 的要求同第 3 条。
Slang 着色器在**首次运行时 JIT 编译**，所以第一次跑训练会慢几分钟，之后走缓存。

## 5. 其余检查清单

- **每个仓库一个独立 venv**（官方 3DGS / Cloud-GS / Vol3DGS 三个仓库都装同名包 `diff_gaussian_rasterization`，混装必坏）；
- **LPIPS 统一**：三边都用 `lpips` 包 + `net='vgg'`。注意 **vgg16 权重首次使用会联网下载**；
  离线/内网机器请预先把 `vgg16-397923af.pth` 放到 `~/.cache/torch/hub/checkpoints/`（可从本机 `C:\Users\17744\.cache\torch\hub\checkpoints\` 拷贝）；
- `numpy 1.26.4` 与 `opencv-python 4.13` 的元数据冲突只是警告（ABI 实测兼容）；万一 `import cv2` 失败，
  改钉 `opencv-python==4.10.0.84`；
- 训练/评测**无需图形界面**（SIBR viewer 不参与）；headless 服务器可直接跑；
- 需要拷贝的数据集：`D:\CloudDatasetZenith`（73 训练 + 36 测试）等，目录内相对路径结构保持不变即可。

## 6. 跨机器对比的注意事项

- **指标可比**（同一数据集、同一 split、同一 LPIPS 实现），但 **FPS/显存数字不可跨机器比较**——论文里报告绝对 FPS 时请注明 GPU 型号；
- 不同 CUDA/驱动版本可能带来极小数值差异，建议同一张表内的所有方法在同一台机器上跑（这正是 `--render_backend slang`（原版 3DGS）与 `slang_volr`（Vol3DGS）同仓库双后端存在的价值：可做最受控对比）。
