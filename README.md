# TextING-pytorch
The code and dataset for ACL2020 paper [Every Document Owns Its Structure: Inductive Text Classification via
Graph Neural Networks](https://www.aclweb.org/anthology/2020.acl-main.31.pdf),implemented in Pytorch.
Some functions are based on [TextGCN](https://github.com/yao8839836/text_gcn).Thank for their work.
The original code implemented in Tensorflow is [TextING](https://github.com/CRIPAC-DIG/TextING).Thank for their work too.
# Comprehensions
[Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks论文理解](https://blog.csdn.net/jokerxsy/article/details/113756400)
# Requirements
- Python 3.6+
- Pytorch 1.7.1(other versions may also work~)
- Scipy 1.5.1
# Usage
Download pre-trained word embeddings `glove.6B.300d.txt` from [here](http://nlp.stanford.edu/data/glove.6B.zip) and unzip to the repository.
Build graphs from the datasets in data/corpus/ as:
```python
python build_graph.py [DATASET] [WINSIZE]
```
Examples:
```python
python build_graph.py R8
```
Provided datasets include `mr`,`ohsumed`,`R8`and`R52`. The default sliding window size is 3.
To use your own dataset, put the text file under `data/corpus/` and the label file under `data/` as other datasets do. Preprocess the text by running `remove_words.py` before building the graphs.
Start training and inference as:
```python
python train.py [--dataset DATASET] [--learning_rate LR]
                [--epochs EPOCHS] [--batch_size BATCHSIZE]
                [--hidden HIDDEN] [--steps STEPS]
                [--dropout DROPOUT] [--weight_decay WD]
```
Examples:
```python
python train.py --dataset R8
```
To reproduce the result, large hidden size and batch size are suggested as long as your memory allows. We report our result based on 96 hidden size with 1 batch. For the sake of memory efficiency, you may change according to your hardware. Program uses cpu by default.

# Reproduced Results
||MR|R8|R52|Ohsumed|
|-|-|-|-|-|
|-|-|0.96345|-|-|


Thank [TextING](https://github.com/CRIPAC-DIG/TextING) again~
