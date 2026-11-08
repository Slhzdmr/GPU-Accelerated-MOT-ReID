# GPU-Accelerated MOT & ReID Pipeline

A lightweight, custom pipeline for Multi-Object Tracking (MOT) and Person Re-Identification (ReID). I built this architecture to solve the CPU-GPU transfer bottlenecks and memory overflow issues I faced during standard tracking tasks.

## Technical Details & Optimizations

* **Batch Inference:** Instead of feeding person crops to the model one by one, the pipeline uses `torch.cat` to batch them into a single tensor before the forward pass. 
* **Fast Cosine Similarity:** Embeddings are L2-normalized (`p=2`) right at the vision engine level. This allows the gallery to match identities using pure hardware-accelerated matrix-vector multiplication (`torch.mv`) instead of slow distance loops.
* **LRU Memory Management:** To prevent RAM/VRAM overflow during long videos, the identity database is built on an `OrderedDict` acting as an LRU (Least Recently Used) Cache. It caps the tracking limit (e.g., 50 people) and automatically evicts the oldest unseen identities.

## Project Structure

* `vision_engine.py`: Handles zero-shot YOLOv8 detection and ResNet-18 feature extraction (modified with `nn.Identity()`).
* `reid_lru_gallery.py`: Pure tensor math for matching embeddings and managing the LRU cache.
* `main.py`: The main loop that reads frames and links the vision engine to the gallery.

## How to run

First, clone the repository and install the basic dependencies:

```bash
git clone https://github.com/Slhzdmr/GPU-Accelerated-MOT-ReID.git
cd GPU-Accelerated-MOT-ReID
pip install -r requirements.txt
```

**Important Note for NVIDIA GPU Users:**
To enable CUDA hardware acceleration, replace the default CPU-only PyTorch installation with the CUDA 12.1 version before running the script:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Run the pipeline:**

```bash
python main.py
```