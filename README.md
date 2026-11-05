<h1 align="center">MARICL</h1>

<p align="center">
  <b>Multi-Agent Residual In-Context Learning</b><br>
  Ask the LLM to debug a model rather than be one.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.22897"><img alt="arXiv 2605.22897" src="https://img.shields.io/badge/arXiv-2605.22897-b31b1b"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <a href="examples/synthetic_data"><img alt="Examples replay with no API key" src="https://img.shields.io/badge/examples-no%20API%20key%20needed-2ea44f"></a>
</p>

A base model makes mistakes. MARICL hands those mistakes to an LLM, which writes a readable formula that fixes them. At prediction time only the formula runs, with no LLM calls.

Code for [*From Residuals to Reasons: LLM-Guided Mechanism Hypotheses from Tabular Data*](https://arxiv.org/abs/2605.22897).

## Results on synthetic data

Claude Opus 5.5 fixes the residuals of a linear model (regression) and a logistic model (classification). It is told nothing about the data.

![Results on synthetic data](examples/synthetic_data/figures/synthetic_results.png)

| | Base model | XGBoost | **MARICL** | Best possible |
|---|---:|---:|---:|---:|
| Regression, test R² | 0.370 | 0.787 | **0.914** | 0.987 |
| Classification, test accuracy | 0.448 | 0.900 | **0.938** | 0.990 |

Means over two seeds. Both examples replay with no API key:

```bash
cd examples/synthetic_data
pip install -r requirements.txt
python agent_step.py --task regression --seed 0
python agent_step.py --task classification --seed 0
```

The [examples README](examples/synthetic_data/README.md) shows the formulas Claude wrote.

## How it works

1. **Fit** a base model, such as linear regression or XGBoost.
2. **Read.** LLM agents look at the examples with the largest errors and say what the model is missing.
3. **Write.** Each idea becomes a sandboxed Python formula.
4. **Refine.** The LLM critiques each formula where it still fails and rewrites it.
5. **Predict** with the base model plus the weighted formulas. No LLM is called.

$$\hat{y}(\mathbf{x}) = f_{\text{ML}}(\mathbf{x}) + \lambda(\mathbf{x}) \sum_{k} \alpha_k(\mathbf{x})\, f_k(\mathbf{x})$$

The formulas are readable, executable hypotheses, not validated mechanisms. Check them with a domain expert before relying on them.

## Use it on your data

Pass in LangChain-style chat models, or the Claude adapter in [`examples/synthetic_data/claude_backend.py`](examples/synthetic_data/claude_backend.py).

**Regression** (scale the target to [0, 1]):

```python
import maricl_core as mc
from maricl_headline import select, maricl_predict

mc.configure_llms(llm, llm_creative)
model = mc.MARICL(task="regression", K=2, kappa=0.3, T=10, tau_fail=0.1)
model.fit(X_train, y_train, base_model=base, feature_names=names,
          feature_descs=descs, domain_context="What this data is about")

chosen, scores, _ = select(model, X_train, y_train, X_val, y_val,
                           base.predict(X_train), base.predict(X_val))
y_hat = maricl_predict(model, X_test, base.predict(X_test), chosen, scores)
```

**Classification** (integer labels 0 to C−1):

```python
from maricl_headline import select_classification, select_beta, maricl_predict_proba

model = mc.MARICL(task="classification", K=2, kappa=0.3, T=10)
model.fit(X_train, y_train, base_model=base, feature_names=names,
          feature_descs=descs, domain_context="What this data is about", n_classes=C)

P_tr, P_va, P_te = (base.predict_proba(X) for X in (X_train, X_val, X_test))
chosen, scores, _ = select_classification(model, X_train, y_train, X_val, y_val, P_tr, P_va)
beta = select_beta(model, X_val, y_val, P_va, chosen, scores)
proba = maricl_predict_proba(model, X_test, P_te, chosen, scores, beta)
```

The `maricl_headline` functions pick the best refinement step on validation data, as in the paper, and make no LLM calls.

## What the paper found

- MARICL improves its base model on **all nine** benchmarks.
- It leads the strongest baseline on **five of eight** real benchmarks and trails TabPFN on the other three. After correction for multiple comparisons, none of the leads is significant.
- About **half** of the gain survives with anonymized features, no domain context, and a small open model.
- Fluent is not correct: with scrambled feature names, MARICL can still tell a convincing but wrong story.

<details>
<summary><b>Abstract</b></summary>

> When software fails, we read the error, form a hypothesis, write a patch, and test it. LLMs now do this for neural networks by editing weights; when a tabular model fails, we usually retune it. Most interpretability methods report which features matter rather than how they combine. Rather than asking an LLM to predict the target over the whole output space, we anchor the prediction with a base model and ask the LLM to debug it. We introduce **Multi-Agent Residual In-Context Learning (MARICL)**, in which LLM agents read high-residual examples, hypothesize the missing structure, and write executable correction terms that are then refined by multi-turn textual gradient optimization. The resulting corrections are readable, executable hypotheses about what the base model misses, rather than validated mechanisms. Across nine benchmarks covering scientific, biomedical, socioeconomic, and synthetic data, MARICL reliably repairs its base model's structured residuals, improving it on every dataset, and on synthetic data absent from pretraining it recovers the hidden structure (two seeds). Against the strongest competing baseline it leads on five of the eight real benchmarks and trails TabPFN on the other three, while adding an inspectable correction that black-box baselines do not provide.

</details>

<details>
<summary><b>Demo notebook (OpenAI)</b></summary>

[`maricl_demo_v1.ipynb`](maricl_demo_v1.ipynb) runs a small regression and classification demo with an OpenAI model. It carries its own copy of the pipeline.

```bash
pip install langchain langchain-openai openai pydantic numpy scikit-learn matplotlib python-dotenv
```

Put `OPENAI_API_KEY=...` in a `.env` file in the project root (never commit it), then run the cells in order. `MAICL_MODEL_NAME` (default `gpt-5.4-mini`) and `OPENAI_REASONING_EFFORT` (default `low`) can be overridden.

</details>

## Repository layout

```
.
├── maricl_core.py           # The MARICL pipeline
├── maricl_headline.py       # Validation selection and the outer gate used in the paper
├── maricl_demo_v1.ipynb     # Demo notebook (OpenAI)
└── examples/
    └── synthetic_data/      # Regression and classification examples with Claude
```

## Citation

```bibtex
@misc{rezaei2026residualsreasonsllmguidedmechanism,
      title={From Residuals to Reasons: LLM-Guided Mechanism Inference from Tabular Data}, 
      author={Mohammad R. Rezaei and Rahul G. Krishnan},
      year={2026},
      eprint={2605.22897},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.22897}, 
}
```

## License

MIT. See [LICENSE](LICENSE).
