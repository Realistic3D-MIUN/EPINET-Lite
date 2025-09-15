# Light Field Disparity Estimation – Author Implementations

![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-Coming%20Soon-blue)

This repository provides the official implementations of our three research works on **light field disparity estimation**.  
All models are implemented in **TensorFlow** and associated Python libraries.

---

## 📄 Implemented Papers

1. **Light-weight Epinet Architecture for Fast Light Field Disparity Estimation**  
   *Ali Hassan, Mårten Sjöström, Tingting Zhang, Karen Egiazarian*  
   IEEE 24th International Workshop on Multimedia Signal Processing (MMSP), 2022  
   [[IEEE Xplore]](https://ieeexplore.ieee.org/document/9949378)

2. **REDARTS: Regressive Differentiable Neural Architecture Search for Exploring Optimal Light Field Disparity Estimation Network**  
   *Ali Hassan, Mårten Sjöström, Tingting Zhang, Karen Egiazarian*  
   IEEE Transactions on Emerging Topics in Computational Intelligence, 2025   
   [[IEEE Xplore]](https://ieeexplore.ieee.org/document/11141437)  

3. **EPINET-Lite: Rethinking Mixed Convolutions for Efficient Light Field Disparity Estimation Network**  
   *Ali Hassan, Tingting Zhang, Karen Egiazarian, Mårten Sjöström*  
   IEEE 27th International Workshop on Multimedia Signal Processing (MMSP), 2025   
   [[Coming Soon]](https://attend.ieee.org/mmsp-2025/)

---

## 🔎 Overview of EPINET-Lite

**EPINET-Lite: Rethinking Mixed Convolutions for Efficient Light Field Disparity Estimation**  

Light field disparity estimation with CNNs often suffers from **high computational cost**, mainly due to large convolutional kernels. While small kernels are efficient, they may fail to capture broader spatial-angular context.  

This work introduces **EPINET-Lite**, a lightweight yet accurate network that integrates an **Optimized Mixed Convolution (MixConv+)** operation into the EPINET architecture. Unlike conventional mixed convolutions that split channels into fixed groups, MixConv+ applies **pointwise + depthwise convolutions in parallel with multiple kernel sizes**, allowing the network to learn multi-scale features more effectively.  

### 🚀 Key Contributions
- **Analysis of convolution operations:** Studied standard, depthwise separable, inverted bottleneck, and mixed convolutions across different kernel sizes.  
- **MixConv+:** Proposed an optimized mixed convolution that improves efficiency while capturing features across multiple receptive fields.  
- **Efficiency:** Achieves up to **4.2× reduction in model complexity** compared to standard convolutions.  
- **Performance:** Demonstrates **improved disparity estimation performance** on the HCI light field benchmark while using **fewer parameters** and maintaining fast inference.  

### 📊 Results
- **76% reduction** in parameters compared to baseline EPINET.  
- Achieves **lowest mean square error (MSE)** among compared methods, while staying competitive on Bad Pixel Ratio metrics.  
- Produces sharper and more consistent disparity maps, even in challenging scenes (e.g., fine textures, occlusions).  

EPINET-Lite balances **accuracy and efficiency**, making it practical for **real-time applications** and deployment on **resource-constrained environments**.  

---

## ⚙️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Realistic3D-MIUN/EPINET-Lite.git
   cd EPINET-Lite
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv lfd-env
   source lfd-env/bin/activate   # Linux/Mac
   lfd-env\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify TensorFlow installation:
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

---

## 🚀 Usage

- Each Python main script corresponds to one of the papers:
  - EPINET-D: `EPINET_D_train.py` and `EPINET_C.py`, `EPINET_CD_train.py`, `EPINET_DC_train.py` - IEEE MMSP 2022
  - LFNASNet: `EPINASNET_A_train.py` and `EPINASNET_B_train.py` - IEEE TECTI 2025
  - EPINET-Lite: `EPINET-Lite_train.py` - IEEE MMSP 2025

- Run training or evaluation scripts as described in their respective subfolders.  
- Example:
  ```bash
  cd EPINET-Lite
  python file_name.py
  ```

---

## 📚 Citation

If you find this repository useful in your research, please cite our works:

```bibtex
@INPROCEEDINGS{hassan2022lightweightepinet,
  author={Hassan, Ali and Sjöström, Mårten and Zhang, Tingting and Egiazarian, Karen},
  booktitle={2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP)}, 
  title={Light-Weight EPINET Architecture for Fast Light Field Disparity Estimation}, 
  year={2022},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/MMSP55362.2022.9949378}}

@ARTICLE{hassan2025redarts,
  author={Hassan, Ali and Sjöström, Mårten and Zhang, Tingting and Egiazarian, Karen},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={REDARTS: Regressive Differentiable Neural Architecture Search for Exploring Optimal Light Field Disparity Estimation Network}, 
  year={2025},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TETCI.2025.3592281}}

@INPROCEEDINGS{hassan2025epinetlite,
  author={Hassan, Ali and Zhang, Tingting and Egiazarian, Karen and Sjöström, Mårten},
  booktitle={2025 IEEE 27th International Workshop on Multimedia Signal Processing (MMSP)}, 
  title={EPINET-Lite: Rethinking Mixed Convolutions for Efficient Light Field Disparity Estimation Network}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/MMSP55362.2022.9949378}}
```

---

## 📬 Contact

For questions, please contact: **Ali Hassan** (ali.hassan@miun.com)
