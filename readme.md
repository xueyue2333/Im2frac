# Im2frac: From Images to Fractures

Convert multi-angle 2-D images into 3-D point clouds and extract fracture surfaces.

---

## 1. Overview

| Stage       | Purpose |
|-------------|---------|
| **im2pcd**  | Images → 3-D point cloud (`.pcd`) |
| **pcd2frac**| 3-D point cloud → fracture surfaces |

---

## 2. Prerequisites

| Tool     | How to Install |
|----------|----------------|
| **COLMAP** | Download from [releases](https://github.com/colmap/colmap/releases) and add the folder containing `COLMAP.exe` to your system `PATH`. |
| **MATLAB** | Required for the fracture-extraction GUI. |

---

## 3. Quick Start

### 3.1 Generate 3-D Point Cloud (Python)

```bash
cd im2pcd
python main.py --init_path ../im2pcd/example_data/data
```

### 3.2 Fracture Extraction (MATLAB)

| Step | Instruction                                                               |
|------|---------------------------------------------------------------------------|
| **Input** | After step 3.1, the generated `.pcd` files are saved in `im2pcd/outputs`. |
| **Launch MATLAB** | Run `pcd2frac/mainFun.m`.                                                 |

Multiple pcd files will be exported. View it using point cloud visualization software such as cloudCompare.

