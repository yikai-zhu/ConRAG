<h1 align="center">ConRAG: Consensus-Driven Evidence Flow for Multi-Hop Question Answering</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.28093">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&style=flat-square" alt="arXiv:2605.28093">
  </a>
  <a href="https://github.com/yikai-zhu/ConRAG">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?logo=github&style=flat-square" alt="GitHub">
  </a>
  <a href="pyproject.toml">
    <img src="https://img.shields.io/badge/Python-3.14%2B-blue?logo=python&style=flat-square" alt="Python 3.14+">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License">
  </a>
</p>

<p align="center">
  <a href="#framework">Framework</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#citation">Citation</a>
</p>

Official implementation of **ConRAG**, a retrieval-augmented generation framework for multi-hop question answering. ConRAG combines text, entity, and relation evidence to select supporting passages, then transfers intermediate answers through explicit query slots to guide subsequent retrieval and reasoning.

## Framework

![ConRAG framework: Connection, Consensus, and Constraint](figures/framework.png)

- **Connection** organizes cross-passage evidence into a knowledge graph while preserving the source passages of entities and relations.
- **Consensus** ranks passages using complementary Text, Entity, and Relation views, with a bounded structural bonus added to text relevance.
- **Constraint** binds intermediate answers to dependent query slots, shares evidence along declared dependencies, and executes independent steps concurrently.

## Quick Start

### 1. Install

```bash
git clone https://github.com/yikai-zhu/ConRAG.git
cd ConRAG

conda create -n conrag python=3.14 -y
conda activate conrag
python -m pip install -e .
```

### 2. Run the example

Set your API key in [`src/conrag/config.py`](src/conrag/config.py).

Run the included example from the repository root:

```bash
conrag --dataset example
```

The first run builds the knowledge graph and vector indices; subsequent runs reuse them. Cached indices are stored in `outputs/<dataset>/`, and predictions, evaluation scores, and logs are saved to `results/<dataset>/<timestamp>/`.

## Citation

If you find this useful in your research, please consider citing:

```bibtex
@article{zhu2026conrag,
  title={ConRAG: Consensus-Driven Evidence Flow for Multi-Hop Question Answering},
  author={Zhu, Yikai and Chen, Kunfeng and Zhong, Qihuang and Liu, Juhua and Du, Bo},
  journal={arXiv preprint arXiv:2605.28093},
  year={2026},
  doi={10.48550/arXiv.2605.28093},
  url={https://arxiv.org/abs/2605.28093}
}
```

## License

The code is released under the [MIT License](LICENSE). Benchmark datasets retain their original licenses.
