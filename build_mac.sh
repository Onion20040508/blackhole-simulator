#!/bin/bash
# ============================================================
#  KERR // BLACK HOLE SANDBOX -- macOS 一键打包脚本
#  用法：把整个 blackhole-sim 文件夹拷到 Mac 上，终端运行：
#      cd blackhole-sim && bash build_mac.sh
#  产物：dist/KerrBlackHole-macos.zip（解压后运行 KerrBlackHole）
#
#  说明：
#  - Mac 没有 CUDA，Taichi 自动走 Metal/Vulkan 后端（帧率低于
#    NVIDIA 显卡，Apple Silicon 可玩，建议把 render scale 调低）
#  - 需要 Python 3.10~3.12（brew install python@3.11）
#  - 未签名应用首次打开：右键 -> 打开，或
#      xattr -dr com.apple.quarantine KerrBlackHole/
# ============================================================
set -e

PY=python3
for cand in python3.11 python3.12 python3.10 python3; do
    if command -v $cand >/dev/null 2>&1; then PY=$cand; break; fi
done
echo "using $($PY --version)"

$PY -m venv .venv_mac
source .venv_mac/bin/activate
pip install --quiet --upgrade pip
pip install --quiet taichi numpy pillow pyinstaller

# 先验证 taichi 在这台 Mac 上能起 GPU 后端
python - <<'EOF'
import taichi as ti
ti.init(arch=ti.gpu)
print("taichi GPU backend OK")
EOF

pyinstaller --noconfirm --noupx --name KerrBlackHole \
    --collect-all taichi \
    --hidden-import PIL --hidden-import PIL.Image \
    --exclude-module matplotlib --exclude-module scipy --exclude-module tkinter \
    --add-data "kerr_raytracer.py:." \
    kerr_raytracer.py

# 快速自检：离线渲染一帧
./dist/KerrBlackHole/KerrBlackHole --still --res 320 180 --steps 150 --out /tmp/kerr_test
echo "self-test render OK"

cp DIST_README.txt dist/KerrBlackHole/使用说明.txt 2>/dev/null || true
cd dist && zip -qr KerrBlackHole-macos.zip KerrBlackHole
echo "=============================================="
echo "done: dist/KerrBlackHole-macos.zip"
echo "=============================================="
