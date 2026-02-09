# EdgeJudge / RepJudge – Full Setup (NVIDIA AGX Xavier)

Step-by-step installation from env create through project dependencies (through point 5).

**Important:** Steps 1 and 2 (exports in `~/.bashrc`) must be done **before** installing PyTorch. The CUDA and conda PATH exports are required so `pip` installs into the right env and `import torch` can find the CUDA libraries.

---

## System

- **Board:** NVIDIA AGX Xavier  
- **JetPack:** 5.0.2 (L4T R35.1)  
- **CUDA:** 11.4 (at `/usr/local/cuda-11.4`)  
- **Python:** 3.8 

---

## 1. CUDA environment 

Add these lines to `~/.bashrc` so every new shell has CUDA and the correct library path. **You must do this before installing PyTorch**; otherwise `import torch` will fail with missing `libcudart` / `libcublas`.

```bash
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
```

Apply in the current terminal (or open a new one):

```bash
source ~/.bashrc
```

*(On JetPack, `libcudart` and `libcublas` are under `targets/aarch64-linux/lib`; without this path, `import torch` can fail with missing libs.)*

---

## 2. Conda environment and PATH in `~/.bashrc`

Create the environment with Python 3.8:

```bash
conda create -n repjudge python=3.8 -y
```

Activate it:

```bash
conda activate repjudge
```

**Add this to `~/.bashrc`** so the active conda env's `pip` and `python` are used first (otherwise `pip install` may install to the wrong place). This block is often added by conda init; if not, add it:

```bash
# After conda init block in ~/.bashrc — use active env's bin first
if [[ -n "$CONDA_PREFIX" ]]; then
  export PATH="$CONDA_PREFIX/bin:$PATH"
fi
```

Then run `source ~/.bashrc` (or open a new terminal) and ensure you're in the `repjudge` env before any `pip install`.


---

## 3. PyTorch (CUDA 11.4, aarch64)

**Before installing:** (1) CUDA and `LD_LIBRARY_PATH` are in `~/.bashrc` and you've run `source ~/.bashrc`, (2) `conda activate repjudge` is active, (3) `~/.bashrc` has the conda `PATH` line above so `which pip` and `which python` point to the env.

Then install the NVIDIA-built wheel:

```bash
pip install torch-1.13.0a0+410ce96a.nv22.12-cp38-cp38-linux_aarch64.whl
```

If the wheel is not in the current directory, use the full path or download from NVIDIA's index (e.g. `jp/v502/pytorch/`) and pass that path to `pip install`.

---

## 4. Verify PyTorch and CUDA

Still with `conda activate repjudge`:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected:

- Version: `1.13.0a0+410ce96a.nv22.12`  
- CUDA available: `True`

---

## 5. Project dependencies (MediaPipe backend)

From the repo root, with `repjudge` activated:

```bash
pip install -r RepJudge/mediapipe/requirements.txt
```

That installs: `mediapipe`, `opencv-python-headless`, `numpy`.

Optional extras (e.g. for scripts that use pandas, plotting, Excel, YAML):

```bash
pip install scipy pandas matplotlib openpyxl PyYAML
```


## mmdeploy on Nvidia AGX Xavier
# Build from source

```bash
cd /media/nvidia-agx-donkey/ssdhome/nvidia-agx-donkey/RepJudge
git clone https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
git checkout v1.3.1
```
# Install MMDeploy python requirements
```bash
pip install -r requirements.txt
```
# If Cmakelist issue
```bash
git submodule update --init --recursive
```
# then
```bash
mkdir build && cd build

cmake .. \
  -DMMDEPLOY_BUILD_SDK=ON \
  -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
  -DMMDEPLOY_BUILD_TENSORRT=ON \
  -DMMDEPLOY_BUILD_TORCHSCRIPT=ON \
  -DMMDEPLOY_TARGET_DEVICES="cuda" \
  -DMMDEPLOY_TARGET_BACKENDS=trt \
  -DTENSORRT_DIR=/usr/lib/aarch64-linux-gnu \
  -DCUDNN_DIR=/usr \
  -DCUB_ROOT_DIR=/media/nvidia-agx-donkey/ssdhome/nvidia-agx-donkey/RepJudge/mmdeploy/third_party/cub \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
make install
```

# Add path
```bash
echo 'export MMDEPLOY_DIR=/media/nvidia-agx-donkey/ssdhome/nvidia-agx-donkey/RepJudge/mmdeploy' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$MMDEPLOY_DIR/build/install/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

```
# Install model converter
```bash
cd /path/mmdeploy
pip install -e .

```
# If backend is not worked for tensorrt
Change the cmakeLists.txt
```bash
ADD after 2nd bottom
if(CUB_ROOT_DIR)
      target_include_directories(${PROJECT_NAME}_obj PRIVATE ${CUB_ROOT_DIR})
endif()
Remove line 26
```
# Tensorrt conversion code
```bash
python tools/deploy.py   /media/nvidia-agx-donkey/ssdhome/nvidia-agx-donkey/RepJudge/mmdeploy/configs/mmpose/custom/pose-detection_tensorrt-192x256.py   /media/nvidia-agx-donkey/ssdhome/nvidia-agx-donkey/RepJudge/mmpose/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-l_8xb256-420e_humanart-256x192.py   https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth   /media/nvidia-agx-donkey/ssdhome/nvidia-agx-donkey/RepJudge/RepJudge_edge/latency/input.jpg   --work-dir work_dirs/rtmpose_trt   --device cuda:0   --dump-info
```
```bash
python tools/deploy.py   /media/nvidia-agx-donkey/ssdhome/nvidia-agx-donkey/RepJudge/mmdeploy/configs/mmdet/detection/detection_tensorrt-fp16_static-640x640.py   /media/nvidia-agx-donkey/ssdhome/nvidia-agx-donkey/RepJudge/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py   https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth   /media/nvidia-agx-donkey/ssdhome/nvidia-agx-donkey/RepJudge/RepJudge_edge/latency/input.jpg   --work-dir work_dirs/rtmdet_trt   --device cuda:0   --dump-info

```
