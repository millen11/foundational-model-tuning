# Foundational Model Tuning on Downstream tasks
## Resources
### Models
[Previous Method](https://huggingface.co/docs/transformers/en/model_doc/patchtst#transformers.PatchTSTForPrediction)
<br>
[Finetuning Method](https://github.com/cambridgeltl/autopeft)
<br>
[Model in Use](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/tree/main/Long-term_Forecasting)

### Model Accessories
[Basics of AdapterHub Adapters](https://docs.adapterhub.ml/contributing/adding_adapters_to_a_model.html)

### Loading Pre-trained model
[Old Method, Currently in use within Repo](https://huggingface.co/transformers/v1.2.0/serialization.html#serialization-best-practices)
<br>
[Newer Method, is supported by AutoPEFT](https://huggingface.co/docs/transformers/serialization)

### Dataset
 [ETT Dataset](https://paperswithcode.com/dataset/ett)
<br>
[ETT Dataset download](https://github.com/zhouhaoyi/ETDataset/tree/main)

### Previous URP student on project
#### Name: Na'Sir Miller
#### Email: millen11@rpi.edu
Feel free to contact me with any questions!

## Why not PatchTST
As you will quickly find out, *PatchTST* is one of the best time-series models currently. Previously, *Summy Liu* (grad-student) worked with the model. I tried working with the model as well, however did not understand how to integrate the model with *AutoPEFT*. I believe you need to implement your own feed foward layers -- among other things -- in order to use it. Additionally, I found the *GPT4TS* model which claims to perform better than models such as *PatchTST*.

As a result of the claims and ease of use, I switched to the *GPT4TS* model, due to its easier *AutoPEFT* integration given its *GPT2* backbone.

## ETT Dataset Overview
An electricity-based time-series dataset with 8 columns, one of which is *OT*. *OT* is the column we want to predict.

* Each month 741 (28 days) - 744 (31 days) data points

## Recommended Train-Test split per dataset documentation
**If you decide to change the default training process**
As given in the [*ETT Dataset* docuementation](https://paperswithcode.com/dataset/ett),
  - train should be about *12 months* of data
  - test should be about *4 months of data*
  - validation should be about *4 months of data*

## Setup Process
### Create a Weights and Biases Account
1. In order to run the AutoPEFT method you will need a Weights and Biases account. Go here to do so: [wandb create account](https://wandb.auth0.com/login?state=hKFo2SB0WXZUc2lvN1BpQVE5ZS1nV0JXOU1xQ1ZZZ0xRRHNvaKFupWxvZ2luo3RpZNkgLWVHQkUxUVlYX1FnLWNzbHAyVlplUU84dDA4Ry1rc0WjY2lk2SBWU001N1VDd1Q5d2JHU3hLdEVER1FISUtBQkhwcHpJdw&client=VSM57UCwT9wbGSxKtEDGQHIKABHppzIw&protocol=oauth2&nonce=QUJfY2hiVVVKdjEzcDJfMQ%3D%3D&redirect_uri=https%3A%2F%2Fapi.wandb.ai%2Foidc%2Fcallback&response_mode=form_post&response_type=id_token&scope=openid%20profile%20email&signup=true)
2. Note your API key, you will need this
3. Run `wandb login`
3. Enter or paste your API key into the console

### Creating AutoPEFT Specific Environment
1. Run `conda env create --name env_name --file autopeft_env.yaml`
* env_name is the name of the environment
2. If the conda environment is not in use yet, run `conda activate env_name`
3. Change into this repository, if not already using `cd timeseries-fm-tuning`
4. Change into the autopeft directory, `cd autopeft`
5. Run the script to ensure the installation worked successfully, `./autopeft_run_one_replicate.sh`
* The script necessary to run the AutoPEFT method on a GPT2 model. You can change this script to fit your needs.
This is where it can get weird. If the program runs without issue igonre the rest of the steps in this section.
6. Unfortunately, I do not remember every error that occured. However, 90% of the issues that occurred were due to dependency issues and the AutoPEFT repo needing some updated imports.
* Since I moved autopeft into this repo you should be fine. However, if you do clone autopeft yourself and what not, this step is something to keep in mind.