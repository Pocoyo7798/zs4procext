# **zs4procext**
[![PyPI - License](https://img.shields.io/pypi/l/langchain-core?style=flat-square)](https://opensource.org/licenses/MIT)
[![GitHub star chart](https://img.shields.io/github/stars/Pocoyo7798/zs4procext?style=flat-square)](https://star-history.com/#Pocoyo7798/zs4procext)
[![Open Issues](https://img.shields.io/github/issues-raw/Pocoyo7798/zs4procext?style=flat-square)](https://github.com/Pocoyo7798/zs4procext/issues)

zs4Procext is tool built to create and run LLM and VLM based extraction pipelines. It allows you to create loops of prompt - model response - parsing for a Q&A aproach for data extraction, accelariting multiple model comparison and model optimization.
## Getting Started

```bash
conda create --name zs4procext python=3.10
conda activate zs4procext
git clone git@github.com:Pocoyo7798/zs4procext.git
cd zs4procext
pip install -e .
```
## Extraction Pipelines

Here is the list of extraction pipelines already created. If the extraction pipeline that you need is not here look at the next chapter to create one or contact us help you. For any pipeline to work there are 4 things you need to pass:

### Paragraph Classification
Paragraph classification consists in identifying as True or False if the paragraph is in a certain class. 
zs4procext-data-visual --help
zs4procext-prompt-template-creator --help
zs4procext-text2actions --help
zs4procext-eval_actions --help
