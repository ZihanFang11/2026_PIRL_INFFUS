
<h2 align="center"> <a href="">Towards Informed Incomplete Multi-View Multi-Label Learning via Prior Knowledge Integration
</a></h2>



# Datasets:

The datasets used can be downloaded from [here](https://drive.google.com/drive/folders/1tI-F6pDHz_CEQYJDjfOLN6Mr70X5-lcW?usp=sharing), 
please download them and put them in datasets to `data`.


##  Quick Start (No Preprocessing Needed)
 ```bash
   python main_sup.py
   ```

##  Custom Pipeline (Optional)
If you want to regenerate everything from scratch:

### Step 0: Download Pretrained LLM
Place your pretrained LLM (e.g., Qwen-2.5-14B-Instruct) inside:
```
pretrain_models/Qwen-2.5-14B-Instruct/
```

### Step 1: Deploy an LLM to conduct pairwise semantic analysis
   ```bash
   python gen_LLM_relation_full.py
   ```

### Step 2: Transformed into a numerical semantic correlation matrix
   ```bash
   python generate_LLM_correlation.py
   ```

### Step 3: Conduct incomplete multi-view multi-label classification
   ```bash
   python main_sup.py
   ```
