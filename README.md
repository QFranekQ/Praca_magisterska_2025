# Classification Models Comparison in sign language (static and dynamic)

This project provides a ready-to-use platform for comparing gesture classification models in sign language. It includes an interactive desktop app built with CustomTkinter, allowing users to test preconfigured models on custom videos or images. Initial tests have already been conducted to verify model integration and performance under challenging visual conditions, using full sign alphabet and 25 words.

---

### Models supported:
- K-NN 
- SVM 
- HMM
- DTW
- CNN
- TDNN

### The platform enables:
- Selecting detection models via a user-friendly configuration panel  
- Loading video (numpy files) or image (csv files)  
- Choosing evaluation metrics from:  
  `precision`, `recall`, `F1` and `detection time`  
- Automatically saving results in both `.json` and `.csv` formats
- Generating graphs (ex. error matrix)
- Real-time visual feedback (if enabled) to inspect model behavior  


---


## 🚀 How to Run

1. Install the required packages (Python 3.12 recommended):
```bash
pip install -r requirements.txt
```

2. Launch the desktop application:
```bash
python gui.py
```

> **Note:**  
> While most file and folder paths are not strictly hardcoded, the application expects a **specific directory structure** to function properly.  
> Please follow the default folder layout provided in the repository, or update the paths in the code if you change it.
