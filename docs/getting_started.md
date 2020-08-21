# Getting Started

## Setup and Deploy Platform

### 1. Create a new GitHub repository

1. Login to your GitHub account, navigate to the [AML Batch Scoring Deployment Template repository](https://github.com/nfmoore/aml-batch-deployment-template) and create a new repository from this template. Use [these](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template) instructions for more details about creating a repository from a template.

### 2. Create and configure a new Azure DevOps Project

1. Create a new project in Azure DevOps. Use [these](https://docs.microsoft.com/en-us/azure/devops/organizations/projects/create-project?view=azure-devops&tabs=preview-page#create-a-project) instructions for more details about creating a project in Azure DevOps.

2. Create an Azure Resource Manager Service connection. This is needed so Azure Devops can connect to your subscription and create/manage resources.

   1. Click on `Project Settings` (found at the bottom-left of the portal).
   2. Click on `Service Connections` (found on the sidebar).
   3. Click `New Service Connection` (found at the top-right of the portal).
   4. Select `Azure Resource Manager` and click `Next`.
   5. Select `Service principal (automatic)` and click `Next`.
   6. Leave `Subscription` as the scope level and select your `subscription` from the dropdown.
   7. In the `Service Connection Name` textbox enter `azure-service-connection`. Note: you can enter another name for your service connection but this will require editing the [`variables.yml`](../.pipelines/templates/variables.yml) file.
   8. Leave the `Allow all pipelines to use this connection` checkbox selected and click `Next`.

   The above steps assume you have `Contributor` or `Owner` access to the subscription or resource group.

   See [these](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/service-endpoints#create-a-service-connection) instructions for more details about creating an Azure Resource Manager Service connection.

3. Create a variable group in Azure DevOps to store values that are reused across multiple pipelines or pipeline stages.

   1. Select the `Library` tab from the `Pipelines` section (found on the sidebar).
   2. Click `+ Variable group` and create a variable group named `aml-deployment-templates`. Note: you can enter another name for your variable group but this will require editing the [`variables.yml`](../.pipelines/templates/variables.yml) file.
   3. The variable group should contain the following required variables:

      | Variable Name          | Suggested Value            |
      | ---------------------- | -------------------------- |
      | `compute_cluster_name` | `cpu-cluster`              |
      | `environment`          | `development`              |
      | `location`             | `australiaeast`            |
      | `namespace`            | `amlplatform`              |
      | `service_connection`   | `azure-service-connection` |

See [these](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/variable-groups?view=azure-devops&tabs=classic#use-a-variable-group) instructions for more details about creating a variable group in Azure DevOps.

### 3. Setup the environment

1. The easiest way to setup the environment is by using the [AML Platform Deployment Template](https://github.com/nfmoore/aml-platform-deployment-template).
2. Alternatively, you can provision the infrastructure manually by:

   1. Creating an Azure Machine Learning Workspace.
   2. Creating a compute cluster with the same name as the `compute_cluster_name` variable in the variable group in Azure DevOps.
   3. Registering a tabular dataset with the name `cardiovascular_disease_train_dataset` using the [Cardiovascular Disease dataset](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset) (which has been adapted from Kaggle) and is available [here](https://github.com/nfmoore/aml-platform-deployment-template/blob/master/data/cardiovascular-disease.csv).
   4. Creating a container in the Azure Blob Storage account created when the Azure Machine Learning Workspace was provisioned. Name this container `batch-score`. If you choose different names ensure you edit the corresponding variables in [`variables.yml`](../.pipelines/variables.yml). Files that you want to be scored can be placed in this container. When triggering the pipleine you will specify this datastore and specifiy the directory where the data resides for scoring and the directory to write the data.

   The above steps assume you have `Contributor` or `Owner` access to the subscription or resource group.

### 4. Deploy the solution

1. In your GitHub repository, edit any variables in [`variables.yml`](../.pipelines/variables.yml) you wish to change for your deployment. Remember to change the values for `service_connection` or `resource_group_name` if you have selected something different in section 2 step 2 above.

2. In your Azure DevOps project, create a pipeline from your repository.

   1. Select the `Pipelines` tab from the `Pipelines` section (found on the sidebar).
   2. Click on `New Pipeline` (found at the top-right of the portal).
   3. Select `GitHub` (authenticate if necessary).
   4. Select your GitHub repository from the list of repositories (authenticate if necessary).
   5. Select `Existing Azure Pipelines YAML File`.
   6. Select the `master` branch and enter `/.pipeline/build-release.yml` as the path or select it from the dropdown and click `Continue`.
   7. Select `Run` from the top-right of the protal to execute the pipeline and deploy the platform.

   See [these](https://docs.microsoft.com/en-us/azure/devops/pipelines/create-first-pipeline) instructions for more details about creating a pipeline in Azure DevOps.

### 5. Run the pipeline

1. In the `batch-score` container in the Azure Blob Storage account upload your data to a directory for scoring. For example, `inputs/2020_01_01`.
2. In the AML studio click `Endpoints` in the sidebar and select `Pipeline Edpoint` in the tab.
3. Click on the pipeline `cardiovascular-disease-batch-score` (remember to select a different name if you changed it in the steps above).
4. Click `Submit` to set up a new pipeline run.
5. In the parameters section change the associated `Datastore` and `Path` for the `input_datapath` and `output_datapath` which correspond to the locations your input data resides (configured in step xxx) and associated output directory. For example, select `batch_score` and `inputs/2020_01_01` for `input_datapath` and `batch_score` and `outputs/2020_01_01` for `output_datapath`.
6. Click `Submit` to run the pipeline. This will create an experiment. Once it is complete your scored data can be found in the `output_datapath` directory. For example, `outputs/2020_01_01` in the `batch-score` container.

### 5. Use data drift

1. When you call your web service endpoint all telemetry is collected and stored in the workspace storage account in a container called `batch-score` (see variable `target_datastore_name`). The path to the `input` data in the blob follows this syntax in this template (see variable `target_dataset_path`): `<input or output>/**/*.csv`. Read more information [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-enable-data-collection).

2. This stored telemetry will be accessable via a dataset. A data drift monitor will be configured using this dataset as a target data and the training dataset as a baseline. This means the web service telemetry will be compared against the training (baseline) dataset to determine if model drift is present.

3. The model data drift detector (as configured in this template) will automatically execute every day to determine if drift is present. Read more about model drift [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-monitor-datasets).
