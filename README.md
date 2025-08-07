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
├── utils/               # Common utilities (logging, metrics, etc.)
│   └── logger.py
├── weights/             # Pretrained weights or checkpoints
│   └── decomp.pth
├── dataset/             # Dataset files or symbolic links
│   ├── train.txt
│   ├── val.txt
│   └── YourDataset/
│       ├── classA/
│       └── classB/
├── train.py             # Entry point for training
├── test.py              # Entry point for evaluation
├── arguments.py         # Command-line argument parser
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

Begin by cloning the repository and setting up the environment:

```bash
git clone https://github.com/yourname/research-base-template.git
cd research-base-template

# Create and activate conda environment
conda create -y -n research-env python=3.8
conda activate research-env

# Install dependencies
pip install torch==1.13.1 torchvision==0.14.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

> ✅ You can modify `requirements.txt` as needed for each specific project.

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

Example training command:

```bash
python train.py --train_data_dir ./dataset/YourDataset --batch_size 32 --epochs 100
```

Example testing command:

```bash
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

## 🧪 Example Use Cases

This base code has been used in:

- Image classification (e.g., Food11, CIFAR-10)
- Domain adaptation (e.g., BRNet-style lighting adaptation)
- Audio classification (e.g., emotion detection)
- Bias mitigation & multimodal alignment

---

## 🛠️ Todo (Optional)

- [ ] Add `src/metrics/` for custom evaluation
- [ ] Add `src/hooks/` for training callbacks
- [ ] Add WandB/CometML logger

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
