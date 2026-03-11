# Cultural Equilibrium - Code Implementation

Source code implementation of CEAF (Cultural Equilibrium Alignment Framework) and baselines for the Cultural Equilibrium project.

## 📁 Directory Structure

```
code/
├── ceaf/                       # CEAF framework implementation
│   └── ceaf_trainer.py         # Main CEAF trainer with Nash equilibrium
│
├── baselines/                  # Baseline implementations
│   ├── rlhf_trainer.py         # RLHF with PPO
│   ├── dpo_trainer.py          # Direct Preference Optimization
│   ├── cultural_finetuning.py  # Cultural adapter-based fine-tuning
│   └── soft_prompt_tuning.py   # Soft prompt tuning for cultures
│
├── utils/                      # Utilities
│   └── data_loader.py          # Dataset loaders (CulturePark, NORMAD)
│
├── evaluation/                 # Evaluation metrics
│   └── metrics.py              # CAS, CBI, Win Rate, Diversity
│
├── configs/                    # Configuration files
│   └── config.yaml             # Main configuration
│
├── main.py                     # Main training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Installation

### System Requirements

- Python 3.9+
- CUDA 11.7+ (for GPU training)
- 4× A100 GPUs (recommended for full training)

### Install dependencies

```bash
cd code/
pip install -r requirements.txt
```

## 📚 Data

### CulturePark Dataset
- **Source:** Wang et al. (2024). EMNLP
- **URL:** https://github.com/CulturePark/CulturePark
- **Format:** JSON with train/validation/test splits

### NORMAD Dataset
- **Source:** Arora et al. (2024). arXiv:2404.12464
- **URL:** https://github.com/saurabh-arora-12/normad
- **Format:** JSONL with cultural annotations

### Data Preparation

```bash
# Download datasets
mkdir -p data/culturepark data/normad

# Place data files in:
# - data/culturepark/train.json
# - data/culturepark/validation.json
# - data/normad/train.jsonl
# - data/normad/test.jsonl
```

## 🎯 Training

### Training CEAF (Proposed)

```bash
python main.py \
  --model ceaf \
  --config configs/config.yaml \
  --culturepark-data data/culturepark \
  --normad-data data/normad \
  --output-dir outputs/ceaf
```

### Training DPO Baseline

```bash
python main.py \
  --model dpo \
  --config configs/config.yaml \
  --culturepark-data data/culturepark \
  --output-dir outputs/dpo
```

### Training RLHF Baseline

```bash
python main.py \
  --model rlhf \
  --config configs/config.yaml \
  --culturepark-data data/culturepark \
  --output-dir outputs/rlhf
```

### Training Cultural Fine-tuning

```bash
python main.py \
  --model cultural_ft \
  --config configs/config.yaml \
  --culturepark-data data/culturepark \
  --output-dir outputs/cultural_ft
```

### Training Soft Prompt Tuning

```bash
python main.py \
  --model soft_prompt \
  --config configs/config.yaml \
  --culturepark-data data/culturepark \
  --output-dir outputs/soft_prompt
```

## 📊 Evaluation

### Evaluate trained models

```bash
python main.py \
  --model ceaf \
  --config configs/config.yaml \
  --culturepark-data data/culturepark \
  --eval-data data/culturepark/validation.json \
  --evaluate-only
```

### Metrics

- **CAS (Cultural Appropriateness Score):** Cultural appropriateness score (1-5, higher is better)
- **CBI (Cultural Bias Index):** Cultural bias index (0-1, lower is better)
- **Win Rate:** Win rate in pairwise comparison (0-1, higher is better)
- **Diversity Score:** Response diversity (0-1, higher is better)

## 📈 Expected Results

Based on the paper, baseline results:

| Method | CAS | CBI | Win Rate | Diversity |
|--------|-----|-----|----------|-----------|
| RLHF | 3.21 | 0.42 | 50.0% | 0.31 |
| DPO | 3.35 | 0.38 | 54.2% | 0.35 |
| Cultural FT | 3.52 | 0.31 | 58.7% | 0.42 |
| Soft Prompt | 3.48 | 0.33 | 57.1% | 0.39 |
| **CEAF (Ours)** | **3.89** | **0.22** | **67.3%** | **0.58** |

## 🔧 Custom Configuration

Edit `configs/config.yaml` to customize:

```yaml
ceaf:
  model_name: "meta-llama/Llama-2-7b-hf"  # Change model
  num_cultural_contexts: 12  # Number of cultural contexts
  learning_rate: 2.0e-05
  batch_size: 4
  equilibrium_iterations: 500  # Number of iterations for Nash equilibrium
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run integration test
python tests/test_ceaf.py
```

## 📝 License

Code belongs to the Cultural Equilibrium research project.

## 📞 Contact

**Author:** Thang Nguyen Xuan  
**Institution:** Hanoi University, Vietnam  

---

**Last Updated:** 2026-03-11
