# Workshop NUC setup

Ubuntu x86_64. Run the steps in order.

## 1. Install VS Code

```bash
sudo apt update
sudo apt install -y wget
wget -O /tmp/code.deb "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64"
sudo apt install -y /tmp/code.deb
```

## 2. Install Miniconda

```bash
wget -O /tmp/Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/Miniconda3.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
source ~/.bashrc
```

- If `conda` asks you to accept the Anaconda Terms of Service:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 3. Install Git LFS and clone

`detection/model/demo.pt` is in Git LFS. Install LFS before cloning.

```bash
sudo apt install -y git git-lfs libgl1 libglib2.0-0
git lfs install
cd ~
git clone https://github.com/HKUArmStrong/dobot-robotic-arm-workshop.git
cd dobot-robotic-arm-workshop
```

- `ls -lh detection/model/demo.pt` should show about 53 MB. A file that starts with `version https://git-lfs.github.com/spec/v1` means LFS did not run. Run `git lfs pull`.

## 4. Create the env and install requirements

Python 3.10 matches the `pyrealsense2` and `pupil-apriltags` wheels.

```bash
conda create -n arm-workshop python=3.10 -y
conda activate arm-workshop
cd ~/dobot-robotic-arm-workshop
pip install -r requirement.txt
```

## 5. Allow this user to open the arm

The Dobot serial port is group `dialout`. Add the current user, then log out and back in.

```bash
sudo usermod -aG dialout "$USER"
```

- After login, `groups` should include `dialout`.
- Plug in the arm and save this station's port:

```bash
conda activate arm-workshop
cd ~/dobot-robotic-arm-workshop
python scripts/check_port.py
```
