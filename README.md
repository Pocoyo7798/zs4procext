# **zs4procext**
[![PyPI - License](https://img.shields.io/pypi/l/langchain-core?style=flat-square)](https://opensource.org/licenses/MIT)
[![GitHub star chart](https://img.shields.io/github/stars/Pocoyo7798/zs4procext?style=flat-square)](https://star-history.com/#Pocoyo7798/zs4procext)
[![Open Issues](https://img.shields.io/github/issues-raw/Pocoyo7798/zs4procext?style=flat-square)](https://github.com/Pocoyo7798/zs4procext/issues)

zs4Procext is tool built to create and run LLM and VLM based extraction pipelines. It allows you to create loops of prompt - model response - parsing for a Q&A aproach for data extraction, accelariting multiple model comparison and model optimization.
## **Getting Started**

```bash
conda create --name zs4procext python=3.10
conda activate zs4procext
git clone git@github.com:Pocoyo7798/zs4procext.git
cd zs4procext
pip install -e .
```
## **Extraction Pipelines**

In this repository you have multiple extraction pipelines available. Every pipeline comes with pre-defined models, model parameters, model template and prompts. Hence you just pass the requirement to run each extraction command. In the case that you want to test different combinations for an existing extraction pipeline we can play with this 4 different arguments:
```bash
--llm_model_name
--llm_model_parameters_path
--prompt_template_path
--prompt_schema_path
```
Right now we support any hugginface model supported by the [vllm](https://docs.vllm.ai/en/v0.7.0/models/supported_models.html) python library. You can define the model you want to run by passing the model name on huggingface or the model folder on your computer on the ```--llm_model_name``` argument. The improve the performance of any pre-trained model you can rely on two things. First you can set the model parameters as temperature and top p passing a file containing the parameters such a [this one](src/zs4procext/resources/vllm_default_params.json) on the ```--llm_model_parameters_path``` argument. The performance of a model is also improved by passing the correct prompt template file. Multiple models already have the correct prompt template associated with their name. You can find the list [here](src/zs4procext/prompt.py) under the ```TEMPLATE_REGISTRY```. If you want to use custom template, download this file, change the content of the "template" key, based on what your need and then pass the new prompt template path on the ```--prompt_template_path``` argument.  Finnaly the prompt is defined on a schema file that can be passed on the ```--prompt_schema_path```. The prompt is divided in 6 parts:

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

You can find multiple explamples of prompt schemas [here](src/zs4procext/resources). Now lets look at all the pipelines you have available now!
### Paragraph Classification
Paragraph classification consists in identifying as True or False if the paragraph is in a certain class. For example you can use it to paragraphas cotnaining experimental procedures from other paragraphs. You can run it in the following way:
```bash
zs4procext-paragraph_classifier --type "n2_physisorption" pathe_to_paragraphs.txt path_to_results.txt
```
The input should be a .txt file containing a different paragraph in each line, while the output is also a .txt file with True or False in each line. Right now there are 3 options for the ```--type``` argument implemented: 'n2_physisorption' to identify nitrogen physorption experimental procedures paragraphs, 'ftir_pyridine' to identity FTIR spectroscopy with adsorbed pyridine experimental procedures pragraphs and 'desilication_dealumination' to identify desilication and dealumination experiemetanl procedures paragraphs. You can run it for other by passing a new prompt on the ```--prompt_schema_path``` argument. You can find a example of a clssifying prompt [here](src/zs4procext/resources/classify_n2_physisorption_schema.json)
### Action Extraction
Action Extraction consists in extracting a sequence of experimental actions that describe the procedure present in the paragraph. To apply run the following line:
```bash
zs4procext-text2actions --actions_type materials  pathe_to_paragraphs.txt path_to_results.txt
```
The input should be a .txt file containing a different paragraph in each line, while the output is also a .txt with a list of actions as python dictionaries. Right now you have 4 options for the ```--actions_type```: 'organic' that uses the the action set defined [here](https://www.nature.com/articles/s41467-020-17266-6), 'pistachio' that uses the same action set from organic with small changes to adapt to the [pistachio dataset](https://www.nextmovesoftware.com/pistachio.html), sac that uses the action set defined [here](https://www.nature.com/articles/s41467-023-43836-5) and "materials" that uses the action set defined [here](https://research.ibm.com/publications/catalysts-synthesis-procedures-extraction-from-synthesis-paragraphs-using-large-language-models). An example of a action extraction prompt is available [here](src/zs4procext/resources/material_synthesis_actions_schema.json). Note that if you want to not use any post processing for your model response you need to set the ```--actions_type``` argument to ```None```, otherwise the post processing of that field will be aplied on the model response. If you want to run the pipeline on a subset of the existing action sets you just need to remove the action that you do not want from the prompt. To augment an existing action set you need to add the new action at the respective action registry [here](src/zs4procext/actions.py). To create a complete new action set you can contact us for further colaborations.

###Sample Finder
The extraction pipeline is Work in Progress

###Table Extraction
Table extraction consists in extracting data from samples or experiments present in tables. For example, for a table containing characterization data all the characterized properties will be extract for each sample, with the names standardized. You can run it using the following line.
```bash
zs4procext-table2data --type catalyst_characterization  image_folder_path path_to_results.txt
```
The input should be a folder containing the table images, while the output is a .txt files containing a list of dictionaries containing the info associated to each samples/experiment. Right now, only one option for ```--type``` argument, that is used to extract data from tables containing characterization data from heteregeneous catalysts. To apply it to other kind of tables, you just need to pass a .json file containing the a new table schema on ```--table_schema_path``` argument. The schema should have a structure like [this](src/zs4procext/randomization.py), where for each type of data you want to extract you need to identify the possible keywords and units associated with it.

###Graphic Extraction

## **Model and Settings Evaluation**

There could be the case that you have a new model or a setting combination that could improve an existing. For this, we have an evaluator for each pipleine to speed up LLM screning and parameter optimization. You can already find multiple model paremeter files [here](src/zs4procext/resources/model_parameters). To do the evaluation you should run the extraction pipeline with the new settings on a reference dataset available [here](src/zs4procext/resources/datasets). Then the following evaluator are availble:
```bash
zs4procext-eval_classifier path_to_reference_file path_to_results_file path_to_evaluation_file.xlxs
zs4procext-eval_actions path_to_reference_file path_to_results_file path_to_evaluation_file.xlxs
zs4procext-eval_graphs path_to_reference_file path_to_results_file path_to_evaluation_file.xlxs
```
You can also find the reference files in [here](src/zs4procext/resources/datasets). Note that each evaluator as a set of paremeters pre defined. If you want to change them run the ```--help``` argument to verify what are the paraemeters available for you to tune.

## **Creating New Pipelines**

The zs4procext tool comes with a bunch of parsers based on regex for you to create your own extraction tool. In this part we will give you an example on how create one. Note all the files used in the example are [here](put_link_here.com) So, imagine that you want to create a tool to identify the alkaline treatment temperature used in the following procedure:

```diff
"The alkaline treatment process was carried out using a 0.2 mol. L-1 NaOH solution on a parent zeolite calined at 673.15 K. In all experiments, 1 g of ZSM-5 zeolite, 100 mL of solution and heating at 338 K under reflux system were used. The duration of the process was limited to 30 and 10 min for conventional electric and microwave (500 W) heating’s, respectively."
```
As you can see even if you look at text you can see that two temeperatures are given. So how can you extract only the alkaline treatment temperature? Lets use LLMs to helps. First we are going to ask to the LLMs to give us the alkaline treatment temperature:

```python
from zs4procext.prompt import PromptFormatter
from zs4procext.llm import ModelLLM

with open("prompt_schema.json", "r") as f:
            prompt_dict = json.load(f)
prompt = PromptFormatter(**prompt_dict)
prompt.model_post_init("phi_mini_4k_template.json")
llm_model = ModelLLM("microsoft/Phi-3-mini-4k-instruct")
llm_model.load_model_parameters("model_parameters.json")
llm_model.vllm_load_model()
final_prompt = prompt.format_prompt(text)
response = llm_model.run_single_prompt(prompt).strip()
print(response)
```
With this is loaded the prompt structure and the model parameters. With this setup you can obtain the model response for multiple paragaphs as well if you need to apply it in large scale. After this the model response should be something like this:

```diff
"For the alkaline treatment 1 g of zeolite was treated with 100 mL 0.2 mol. L-1 NaOH at 338 K"
```

Now the alkaline tratment tempereture value is isolated from other temperatures, however other conditions are also present. To solve it we will use a parameters parser to identify only the temperature and convert it to degrees celsius:

```python
from zs4procext.parser import ParametersParser
temperature_parser = ParametersParser(
            convert_units=True,
            time=False,
            temperature=True,
            pressure=False,
            amount=False,
            atmosphere=False,
            size=False
        )
temperature = conditions_parser.get_parameters(response)["temperature"]
print(tmperature)
```

With this we obtain the temperature in degree celsius:

```diff
["64.85 °C"]
```

Obviously different extraction pipelines will require different parser or a combiantion of them. Hence here it is a table containing all the parser available right and a description on how to use them:

| Parser    | Description |
| -------- | ------- |
| ParametersParser  | $250    |
