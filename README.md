## Robust Long-Term Object Tracking via Improved Discriminative Model Prediction (RLT-DIMP)


----
### Demonstration video


will be uploaded soon.

---

### Abstract

We propose an improved discriminative model prediction method for robust long-term tracking based on a pre-trained short-term tracker. The baseline pre-trained short-term tracker is SuperDiMP which combines the bounding-box regressor of PrDiMP with the standard DiMP classifier. Our tracker RLT-DiMP improves SuperDiMP in the following three aspects: (1) Uncertainty reduction using random erasing: To make our model robust, we exploit an agreement from multiple images after erasing ramdom small rectangular areas as a certainty. And then, we correct the tracking state of our model accordingly. (2) Random search with spatio-temporal constraints: we propose a robust random search method with a score penalty applied to prevent the problem of sudden detection at a distance. (3) Background augmentation for more discriminative feature learning: We augment various backgrounds that are not included in the search area to train a more robust model in the background clutter. In experiments on the VOT-LT2020 benchmark dataset, the proposed method achieves comparable performance to the state-of-the-art long-term trackers.

---
### Prerequisites

will be uploaded soon.






---
### Contact

If you have any questions, please feel free to contact seokeon@kaist.ac.kr

---

### Acknowledgments

- The code is based on the PyTorch implementation of the [DiMP-family](https://github.com/visionml/pytracking). 
- This work was done while the first author was a visiting researcher at CMU. 
- This work was supported in part through NSF grant IIS-1650994, the financial assistance award 60NANB17D156 from U.S. Department of Commerce, National Institute of Standards and Technology (NIST) and by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC) contract number D17PC0034. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copy-right annotation/herein. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as representing the official policies or endorsements, either expressed or implied, of NIST, IARPA, NSF, DOI/IBC, or the U.S. Government.

---

### Citation 

```
@InProceedings{Choi2020,
  author = {Choi, Seokeon and Lee, Junhyun and Lee, Yunsung and Hauptmann, Alexander},
  title = {Robust Long-Term Object Tracking via Improved Discriminative Model Prediction},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={0--0},
  year={2020}
}
```



## Reference


- [1]

- [2]
