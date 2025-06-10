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
```bash
--llm_model_name
--llm_model_parameters_path
--prompt_template_path
--prompt_schema_path
```
Right now we support any hugginface model supported by the [vllm](https://docs.vllm.ai/en/v0.7.0/models/supported_models.html) python library. You can define the model you want to run by passing the model name on huggingface or the model folder on your computer on the ```--llm_model_name``` argument. The improve the performance of any pre-trained model you can rely on two things. First you can set the model parameters as temperature and top p passing a file containing the parameters such a [this one](src/zs4procext/resources/vllm_default_params.json) on the ```--llm_model_parameters_path``` argument. The performance of a model is also improved by passing the correct prompt template file. Multiple models already have the correct prompt template associated with their name. You can find the list [here](src/zs4procext/prompt.py) under the ```TEMPLATE_REGISTRY```. If you want to use custom template, download this file, change the content of the "template" key, based on what your need and then pass the new prompt template path on the ```--prompt_template_path``` argument.  Finnaly the prompt is defined on a schema file that can be passed on the ```--prompt_template_path```. The prompt is divided in 6 parts:

* **Expertise**: Define the closest field related to your problem.
* **Initialization**: Give general tips for the LLM answer.
* **Objective**: Problem definition and indentification of what the LLM should do.
* **Definitions**: Define concepts to be used by the LLM.
  * **Initialization**:Indicate for what to use the definitions.
  * **Concepts**:Indicate the concept name and describe it.
* **Answer Schema**: Define the answer format of the LLM.
  * **Initialization**:Indicate the part of the response that should follow the format and describe it.
  * **Format**:Give an example of the format.
* **Conclusion**: End your prompt giving the last details and take the opportunity to be kind.
  
### Paragraph Classification
Paragraph classification consists in identifying as True or False if the paragraph is in a certain class. 
zs4procext-data-visual --help
zs4procext-prompt-template-creator --help
zs4procext-text2actions --help
zs4procext-eval_actions --help
