

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
    <a href="">Models</a>
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
* [ ] Release checkpoint

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

