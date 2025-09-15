# Light Field Disparity Estimation – Author Implementations

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

## ⚙️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/username/lightfield-disparity.git
   cd lightfield-disparity
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

- Each python main script corresponds to one of the papers:
  - `EPINET_C.py`, `EPINET_CD_train.py`, `EPINET_DC_train.py`, , `EPINET_D_train.py` from Light-Weight EPINET - IEEE MMSP 2022
  - `EPINASNET_A_train.py` and `EPINASNET_A_train.py` from REDARTS - IEEE TECTI 2025
  - `EPINET-Lite_train.py` from IEEE MMSP 2025

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
@inproceedings{hassan2022lightweightepinet, 
  title={Light-weight epinet architecture for fast light field disparity estimation},
  author={Hassan, Ali and Sj{"o}str{"o}m, M{a}rten and Zhang, Tingting and Egiazarian, Karen},
  booktitle={2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}

@article{hassan2025redarts,
  title={REDARTS: Regressive Differentiable Neural Architecture Search for Exploring Optimal Light Field Disparity Estimation Network},
  author={Hassan, Ali and Sj{"o}str{"o}m, M{a}rten and Zhang, Tingting and Egiazarian, Karen},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence},
  year={2025},
  publisher={IEEE}
}

@article{hassan2025epinetlite, 
  title={EPINET-Lite: Rethinking Mixed Convolutions for Efficient Light Field Disparity Estimation Network}, 
  author={Hassan, Ali and Zhang, Tingting and Egiazarian, Karen and Sj{"o}str{"o}m, M{a}rten}, 
  booktitle={2025 IEEE 27th International Workshop on Multimedia Signal Processing (MMSP)}, 
  year={2025},
  publisher={IEEE}
}
```

---

## 📬 Contact

For questions, please contact: **Ali Hassan** (ali.hassan@miun.com)
