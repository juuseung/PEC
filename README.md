# Fine-Tuning Persona-Based Empathetic Conversational Models (PEC)
This repository is a fork of [PEC](https://github.com/zhongpeixiang/PEC).
This project documents my experiments with fine-tuning the PEC model and analyzing its performance.

The original PEC model is described in the following paper: "[Towards Persona-Based Empathetic Conversational Models](https://arxiv.org/abs/2004.12316)" (EMNLP 2020) by Zhong Peixiang, et al.

The dataset used for this repository is the [PEC datasets](https://huggingface.co/datasets/peixiang/pec) available on Huggingface.

---

## Fine-Tuning Overview

### Model Architecture
The experiments focused solely on the **CoBERT** model from [PEC](https://github.com/zhongpeixiang/PEC) due to hardware constraints, 
although the initial plan was to include comparisons with GPT-2 and RoBERTa. 
Frequent crashes caused by GPU and memory limitations prevented the completion of 
experiments with other models.

### Environment Setup
Set up dependencies as per the original repository ([PEC](https://github.com/zhongpeixiang/PEC))'s instructions: the code depends on PyTorch (>=v1.0) and [transformers](https://github.com/huggingface/transformers) (>=v2.3).

### Adjustments Made
To accommodate the available hardware and ensure efficient training:
1. **Learning Rate**: Adjusted from `2e-5` to `5e-5` to optimize performance.

2. **Maximum Sequence Length**: Reduced the [paper](https://arxiv.org/abs/2004.12316)'s default value of `256` tokens to `128` tokens for memory efficiency.  

3. **Early Stopping**: Implemented to prevent overfitting and reduce unnecessary computation.

One of the primary challenges was the computational cost of processing 
large datasets, which led to memory issues on my available hardware. 
GPU resources were also limited, making it necessary to work with subsets of the dataset.

### Dataset Subsets
Instead of using the full PEC dataset, I worked with subsets to train the model efficiently within hardware constraints. 
The following configurations were used:  

1. **Subset 1**:
    - Training: 500 samples
    - Validation: 100 samples
    - Teesting: 100 samples

2. **Subset 2**:
    - Training: 100 samples
    - Validation: 20 samples
    - Teesting: 20 samples


These subsets allowed for experimentation with smaller datasets while maintaining meaningful results.

---

## Training
- **The original PEC model**

    ```python CoBERT.py --config CoBERT_config.json```

- **The fine-tuned PEC model**

    ```python CoBERT_finetuned.py --config CoBERT_finetuned_config.json```

## Evaluation
Set test_mode=1 and load_model_path to a saved model in CoBERT_config.json, and then run

- **The original PEC model**

    ```python CoBERT.py --config CoBERT_config.json```

- **The fine-tuned PEC model**

    ```python CoBERT_finetuned.py --config CoBERT_finetuned_config.json```

---

## Challenges
### Hardware Limitations
1. Memory issues while processing large datasets.

2. Frequent crashes during experiments with models like GPT-2 and RoBERTa due to GPU constraints.

### Adjustments to Overcome Limitations
1. Used subsets of the PEC dataset instead of the full dataset.

2. Focused solely on CoBERT to reduce computational overhead.

3. Optimized workflow with early stopping and adjusted hyperparameters.

---

## Results and Observations
### Loss Comparison
The loss comparison includes three domains: **all**, **happy**, and **offmychest**.  

**Observations**:
- All fine-tuned models achieved a reduction in loss, demonstrating 
effective learning across all subsets and domains. 

### MRR Comparison
The Mean Reciprocal Rank (MRR) measures the average inverse rank of the first correct response:  
- **Higher MRR values** indicate better performance, as they suggest the model ranks correct responses higher on average.  

**Observations**:  
- Only half of the fine-tuned models outperformed the baseline CoBERT in MRR.  
- These results highlight CoBERT’s potential for persona-based tasks but also reveal variability in performance.  

### Analysis
The mixed MRR performance suggests the need for further optimization.  
- **Key Limitation**: The use of dataset subsets likely contributed to variability in downstream performance metrics.  

Despite constraints in hardware and dataset size:  
- This work demonstrates the effectiveness of CoBERT for persona-based empathetic response generation.  
- It underscores CoBERT's potential for persona-driven conversational AI tasks.  

---

## Future Work
1. **Dataset Expansion**: Train on larger datasets to improve downstream performance metrics like MRR.

2. **Model Exploration**: Experiment with other models, such as GPT or RoBERTa, on more robust computational resources.

3. **Optimization**: Investigate advanced optimization techniques to enhance fine-tuning.

## Acknowledgments  
This repository is a fork of [PEC](https://github.com/zhongpeixiang/PEC), and credit for the original implementation goes to the authors of the PEC project.

## License
This project is licensed under the GNU GENERAL PUBLIC LICENSE. See the [LICENSE](LICENSE) file for details.