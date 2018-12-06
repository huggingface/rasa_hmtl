# RASA wrapper for HMTL (Hierarchical Multi-Task Learning)
## 🌊 A State-of-the-Art neural network model for several NLP tasks based on PyTorch and AllenNLP
---


```
@article{sanh2018hmtl,
  title={A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks},
  author={Sanh, Victor and Wolf, Thomas and Ruder, Sebastian},
  journal={arXiv preprint arXiv:1811.06031},
  year={2018}
}
```

⚠ Work in progress, this has not been thoroughly tested. ⚠

Main repo: https://huggingface.co/hmtl/
Demo: https://huggingface.co/hmtl/

This code sample demonstrates how to use `rasa_nlu`'s `Component` mechanism to integrate the tasks results from `HMTL`: 
- Named Entity Recoginition
- Entity Mention Detection
- Relation Extraction
- Coreference Resolution
