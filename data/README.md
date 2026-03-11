# Cultural Equilibrium - Datasets

Thư mục chứa các datasets sử dụng cho dự án Cultural Equilibrium.

## 📊 Tổng quan

| Dataset | Kích thước | Số samples | Trạng thái |
|---------|------------|------------|------------|
| **CulturePark** | 7.2 MB | 41,000+ | ✅ Đã tải về |
| **NORMAD** | 46 MB | 2,630+ | ⚠️ Cần giải mã |

---

## 1. CulturePark Dataset

**Reference:** Wang et al. (2024). CulturePark: Boosting Cross-cultural Understanding in Large Language Models. NeurIPS 2024.

**GitHub:** https://github.com/Scarelette/CulturePark  
**arXiv:** https://arxiv.org/abs/2405.15145

### 📁 Cấu trúc dữ liệu

```
data/culturepark/
├── data/
│   ├── Chinese/          # Dữ liệu văn hóa Trung Quốc
│   ├── Arabic/           # Dữ liệu văn hóa Ả Rập
│   ├── German/           # Dữ liệu văn hóa Đức
│   ├── Korean/           # Dữ liệu văn hóa Hàn Quốc
│   ├── Spanish/          # Dữ liệu văn hóa Tây Ban Nha
│   ├── Portuguese/       # Dữ liệu văn hóa Bồ Đào Nha
│   ├── Bengali/          # Dữ liệu văn hóa Bengal
│   ├── Turkish/          # Dữ liệu văn hóa Thổ Nhĩ Kỳ
│   ├── WVQ.csv           # World Values Questionnaire
│   └── WVQ.jsonl         # WVQ format JSONL
├── data_process.py       # Script xử lý dữ liệu
├── main.py               # Script generation chính
└── README.md             # Documentation
```

### 📍 Các vùng văn hóa (12 dimensions)

Dataset bao gồm dữ liệu từ 8+ quốc gia/vùng lãnh thổ, mapping theo Hofstede's cultural dimensions:

1. **Power Distance** - Khoảng cách quyền lực
2. **Individualism vs Collectivism** - Cá nhân vs Tập thể
3. **Masculinity vs Femininity** - Nam tính vs Nữ tính
4. **Uncertainty Avoidance** - Tránh né rủi ro
5. **Long-term Orientation** - Định hướng dài hạn
6. **Indulgence vs Restraint** - Khoan dung vs Kiềm chế

### 🔧 Sử dụng

```python
# Load CulturePark data
import pandas as pd

# Load WVQ data
wvq_df = pd.read_csv('data/culturepark/data/WVQ.csv')
wvq_jsonl = pd.read_json('data/culturepark/data/WVQ.jsonl', lines=True)

# Load cultural-specific data
china_df = pd.read_csv('data/culturepark/data/Chinese/China.csv')
germany_df = pd.read_csv('data/culturepark/data/Germany/Germany.csv')
```

---

## 2. NORMAD Dataset

**Reference:** Arora et al. (2024). NORMAD: Norms and Morals Across Dialects for Sociocultural Alignment Evaluation. arXiv:2404.12464

**GitHub:** https://github.com/Akhila-Yerukola/NormAd  
**HuggingFace:** https://huggingface.co/datasets/akhilayerukola/NormAd

### 📁 Cấu trúc dữ liệu

```
data/normad/
├── data_and_heval/
│   ├── datasets.zip.enc      # Dataset chính (encrypted)
│   ├── human_eval_inhouse/   # Human evaluation (in-house)
│   └── human_eval_mturk/     # Human evaluation (MTurk)
├── story_prompts/            # Prompts cho story generation
├── conf/                     # Configuration files
├── src/                      # Source code
└── README.md                 # Documentation
```

### ⚠️ Lưu ý quan trọng

Dataset chính (`datasets.zip.enc`) **được mã hóa**. Để truy cập:

1. **Liên hệ authors:** Email cho Akhila Yerukola (akhila@seas.upenn.edu) để xin key giải mã
2. **Sử dụng HuggingFace:** Load trực tiếp từ HuggingFace datasets:
   ```python
   from datasets import load_dataset
   normad = load_dataset("akhilayerukola/NormAd", split="train")
   ```
3. **Sử dụng GitHub data:** Một số data samples có sẵn trong `story_prompts/` và `src/analysis/`

### 📊 Dataset Statistics

- **2,630+ samples** (train split trên HuggingFace)
- **75 countries** được cover
- **305 cultural backgrounds**
- **20 cultural sub-axes**

### 🔧 Sử dụng (HuggingFace)

```python
from datasets import load_dataset

# Load từ HuggingFace
normad = load_dataset("akhilayerukola/NormAd", split="train")

# Xem columns
print(normad.column_names)
# ['ID', 'Country', 'Background', 'Axis', 'Subaxis', 
#  'Value', 'Rule-of-Thumb', 'Story', 'Explanation', 'Gold Label']

# Sample
sample = normad[0]
print(f"Country: {sample['Country']}")
print(f"Story: {sample['Story']}")
print(f"Label: {sample['Gold Label']}")
```

---

## 3. Kết hợp Datasets

Để tạo dataset training cho Cultural Equilibrium:

```python
import pandas as pd
from pathlib import Path

# Load CulturePark
culturepark_data = []
for csv_file in Path('data/culturepark/data').rglob('*.csv'):
    df = pd.read_csv(csv_file)
    df['source'] = 'culturepark'
    culturepark_data.append(df)
culturepark_df = pd.concat(culturepark_data, ignore_index=True)

# Load NORMAD (từ HuggingFace)
from datasets import load_dataset
normad = load_dataset("akhilayerukola/NormAd", split="train")
normad_df = normad.to_pandas()
normad_df['source'] = 'normad'

# Combine
combined_df = pd.concat([culturepark_df, normad_df], ignore_index=True)
print(f"Total samples: {len(combined_df)}")
```

---

## 📝 Hướng dẫn tải về (cho người dùng sau)

### CulturePark

```bash
cd data/culturepark
git clone https://github.com/Scarelette/CulturePark.git .
```

### NORMAD

**Option 1: Từ GitHub**
```bash
cd data/normad
git clone https://github.com/Akhila-Yerukola/NormAd.git .
# Lưu ý: datasets.zip.enc cần key giải mã từ authors
```

**Option 2: Từ HuggingFace (recommended)**
```python
from datasets import load_dataset
normad = load_dataset("akhilayerukola/NormAd", split="train")
normad.to_json("data/normad/normad_train.json")
```

---

## 📊 Statistics (tổng hợp)

| Metric | CulturePark | NORMAD | Total |
|--------|-------------|--------|-------|
| Total samples | 41,000+ | 2,630 | ~43,630 |
| Cultural regions | 8+ | 75 countries | 83+ |
| Languages | 8 | 1 (English) | 8 |
| Format | CSV, JSONL | JSON, CSV | - |
| License | MIT | CC-BY-4.0 | - |

---

## 🔗 Links

- **CulturePark:**
  - GitHub: https://github.com/Scarelette/CulturePark
  - Paper: https://arxiv.org/abs/2405.15145
  - NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2024/hash/77f089cd16dbc36ddd1caeb18446fbdd-Abstract-Conference.html

- **NORMAD:**
  - GitHub: https://github.com/Akhila-Yerukola/NormAd
  - HuggingFace: https://huggingface.co/datasets/akhilayerukola/NormAd
  - Paper: https://arxiv.org/abs/2404.12464

---

## 📞 Contact

**Author:** Thang Nguyen Xuan  
**Institution:** Hanoi University, Vietnam  
**Email:** nxthang@hanu.edu.vn

**Last Updated:** 2026-03-11
