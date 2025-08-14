# 🧠 Research Base Template

> A modular and extensible codebase for deep learning research.  
> Easily adaptable for various tasks

This repository provides a clean and reusable foundation for a wide range of research projects in computer vision, audio, and multimodal learning.  
It is designed to support **rapid prototyping**, **structured experimentation**, and **modular customization**.

---

## 📁 Folder Structure

Organize the folders as:

```
.
├── config/              # Python configs for each experiment
│   └── config.py
├── src/                 # Core logic
│   ├── data/            # Dataset, collators, transforms, dataloader
│   │   ├── dataset.py
│   │   ├── loader.py
│   │   ├── mapping.py   # label mapping
│   │   └── collator.py
│   ├── models/          # Model definitions 
│   │   └── model.py
│   ├── trainer/         # Training and evaluation logic based on trainer
│   │   ├── basetrainer.py
│   │   └── maintrainer.py # define your own Trainer based on basetrainer.py
│   └── utils/               # Common utilities (logging, metrics, etc.)
|
├── dataset/             # Dataset files or symbolic links
│   ├── train.txt
│   ├── val.txt
│   └── YourDataset/
│       ├── classA/
│       └── classB/
├── checkpoints/         # Auto-generated when training
├── log/                 # Auto-generated when running arguments.py or related code
├── train.py             # Entry point for training
├── test.py              # Entry point for evaluation
├── arguments.py         # Command-line argument parser
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

Begin by cloning the repository and setting up the environment:

### Option 1: Using UV (Recommended)

```bash
git clone https://github.com/iontail/basecode.git
cd [your directory]

# Install UV if not already installed
pip install uv

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Option 2: Using Conda + Pip

```bash
git clone https://github.com/iontail/basecode.git
cd [your directory]

# Create and activate conda environment
conda create -y -n basecode python=3.12
conda activate basecode

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> ✅ You can modify `requirements.txt` (for pip/conda) or `pyproject.toml` (for uv) as needed for each specific project.

---

## ⚙️ How to Customize for Your Research

Depending on your task (e.g., classification, detection, multimodal), you should mainly customize the following:

| File                        | Purpose                                  |
|----------------------------|------------------------------------------|
| `src/models/model.py`      | Define your model architecture           |
| `src/data/dataset.py`      | Load and preprocess your dataset         |
| `src/data/collator.py`     | Customize batch structure if needed      |
| `src/trainer/maintrainer.py` | Write task-specific training logic     |
| `config/config.py` or `arguments.py` | Manage hyperparameters and options |

---

## 🏁 Training & Evaluation

### Using UV

```bash
# Training
uv run python train.py --train_data_dir ./dataset/YourDataset --batch_size 32 --epochs 100

# Testing
uv run python test.py --weights ./weights/your_model.pth
```

### Using Traditional Python

```bash
# Training
python train.py --train_data_dir ./dataset/YourDataset --batch_size 32 --epochs 100

# Testing
python test.py --weights ./weights/your_model.pth
```

---

## 📌 Features

- ✅ Clean modular structure
- ✅ Custom collate function support
- ✅ Easy integration of config files or CLI arguments
- ✅ Supports PyTorch native checkpointing and logging
- ✅ Easily extensible to multi-GPU or distributed training

---

## 🛠️ Todo (Optional)

- [ ] modify `src/trainer/basetrainer.py` for custom trainer

---

## 📄 License

This project is open to everyone for research purposes.
