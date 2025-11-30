

<p align="center">
<h1 align="center">📌AStF</h1>
<h3 align="center">Motion Style Transfer via Adaptive Statistics Fusor</h3>
<h4 align="center">ACM Multimedia 2025 (ACMMM'25)</h4>
</p>

<p align="center">
  <p align="center">
    <a href="mailto:hmc@stu.xidian.edu.cn">Hanmo Chen</a><sup>1*</sup>
    ·
    <a href="mailto:chx@stu.xidian.edu.cn">Chenghao Xu</a><sup>1*</sup>
    ·
    <a href="mailto:yanjiexi@xidian.edu.cn">Jiexi Yan</a><sup>1†</sup>
    ·
    <a href="mailto:chdeng@mail.xidian.edu.cn">Cheng Deng</a><sup>1†</sup>
    <br>
    <sup>1</sup>Xidian University
  </p>

  <h3 align="center">
    <a href="https://arxiv.org/abs/2511.04192">Paper</a> |
    <a href="https://github.com/CHMimilanlan/AStFResultGallery">Results Gallery</a> |
    <a href="https://huggingface.co/MilanMagik/AStF">Models</a>
  </h3>
</p>

---

## 📝 Abstract
Human motion style transfer allows characters to appear less rigid
ity and more realism with specific style. Traditional arbitrary image
 style transfer typically process mean and variance which is proved
 effective. Meanwhile,similarmethodshavebeenadaptedformotion
 style transfer. However, due to the fundamental differences between
 images and motion, relying on mean and variance is insufficient
 to fully capture the complex dynamic patterns and spatiotemporal
 coherence properties of motion data. Building upon this, our key
 insight is to bring two more coefficient, skewness and kurtosis,
 into the analysis of motion style. Specifically, we propose a novel 
Adaptive Statistics Fusor (AStF) which consists of Style Disentan
glement Module (SDM) and High-Order Multi-Statistics Attention
 (HOS-Attn). We trained our AStF in conjunction with a Motion
 Consistency Regularization (MCR) discriminator. Experimental re
sults show that, by providing a more comprehensive model of the
 spatiotemporal statistical patterns inherent in dynamic styles, our
 proposed AStF shows proficiency superiority in motion style trans
fers over state-of-the-arts.

---


## 🔥 News

* **[2025.07] Our paper is accepted by ACM Multimedia 2025 (ACMMM'25), Dublin! 🎉**


---

## 🗂 TODO

* [ ] Release train and test code
* [ ] Release Xia checkpoint
* [ ] Release BFA checkpoint

---

## 🔥 Pretrained Models

We release the pretrained ASTF models on **HuggingFace**.
You can download them using either the `huggingface-cli` or by visiting the model page directly.

### **🔗 HuggingFace Repository**
👉 [https://huggingface.co/MilanMagik/AStF](https://huggingface.co/MilanMagik/AStF)

### **📥 Download via CLI**
To download the entire pretrained model package into a local directory named `AStF`, run:

```bash
huggingface-cli download MilanMagik/AStF --local-dir AStF
```
---

## 🛠 Installation

We train and test our AStF on the following environment.
```
Python >= 3.10
CUDA >= 11.6
```

Create a conda environment and install dependencies:

```bash
git clone https://github.com/CHMimilanlan/AStF.git
cd AStF
conda create -n astf python=3.10
conda activate astf
pip install -r requirements.txt
```

---

## 📦 Data Preparation

Our ASTF model is evaluated on **both the Xia dataset and the BFA dataset**.
Below we provide detailed instructions for preparing Xia.
The BFA preprocessing pipeline will be released soon.

---

### 📘 Xia Dataset

### **1. Download the Raw Motion Data**

Download the archive **`mocap_xia.zip`** and place it under:

```
./data/
```

You can obtain the dataset from the official source:

🔗 [https://deepmotionediting.github.io/style_transfer](https://deepmotionediting.github.io/style_transfer)

---

### **2. Unzip the Dataset**

```bash
cd ./data
unzip mocap_xia.zip
```

This extracts the raw `.bvh` motion files required for preprocessing.

---

### **3. Preprocess the Raw Motions**

Run the preprocessing script:

```bash
python xia_preprocess.py
```


After preprocessing, the following directories will be created:

```
preprocessed_xia/          # Processed training data + style/content distributions
preprocessed_xia_test/     # Processed testing sequences
preprocessed_xia_gt/       # Ground-truth training sequences for evaluation
```

These folders are required for **training** and **evaluation** of ASTF on Xia.

---

### 📙 BFA Dataset (Coming Soon)

Our ASTF model is also evaluated on the **BFA dataset**, following the protocol described in our ACMMM’25 paper.

The BFA data preparation scripts will be released soon.
We are currently organizing the preprocessing pipeline, and we will update the repository with BFA support shortly.

---






## 🔧 Train Script

Refer to arguments.py for detailed arguments.

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --save_path results
```
The result checkpoint will be saved in results directory.

---



## 🧪 Evaluation Script

```bash
CUDA_VISIBLE_DEVICES=1 python eval_all.py --model_path pretrained/astf.pth --cntcls_path pretrained/cntcls.pth --stycls_path pretrained/stycls.pth
```
---



## 🙏 Acknowledgements

This codebase is built on top of the open-source implementation of 

* [MoST](https://github.com/Boeun-Kim/MoST)
* [GenMoStyle](https://github.com/Murrol/GenMoStyle-code)
* [Aberman et. al.](https://github.com/DeepMotionEditing/deep-motion-editing)
* [Park et. al.](https://github.com/soomean/Diverse-Motion-Stylization)
---

## 📚 Citation

If you find our work useful, please cite our ACMMM 2025 paper:
```bibtex
@inproceedings{chen2025astf,
  title={AStF: Motion Style Tranfer via Adaptive Statistics Fusor},
  author={Chen, Hanmo and Xu, Chenghao and Yan, Jiexi and Deng, Cheng},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={5557--5566},
  year={2025}
}
```

