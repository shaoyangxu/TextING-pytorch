# TextING-pytorch
The code and dataset for ACL2020 paper [Every Document Owns Its Structure: Inductive Text Classification via
Graph Neural Networks](https://www.aclweb.org/anthology/2020.acl-main.31.pdf),implemented in Pytorch.
Some functions are based on TextGCN.Thank for their work.
The original code implemented in Tensorflow is [TextING](https://github.com/CRIPAC-DIG/TextING).Thank for their work too.
# Requirements
- Python 3.6+
- Pytorch 1.7.1(other versions may also work~)
- Scipy 1.5.1
# Usage
Download pre-trained word embeddings `glove.6B.300d.txt` from [here](http://nlp.stanford.edu/data/glove.6B.zip) and unzip to the repository.
Build graphs from the datasets in data/corpus/ as:
```python
hello
```
