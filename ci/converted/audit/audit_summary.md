# Audit summary

Summary for [Azure DevOps instance](https://dev.azure.com/ExaScience/smurff/_build)

- GitHub Actions Importer version: **1.3.22039 (9432caa2d13df57db8619b9b253cdf994da6625c)**
- Performed at: **6/3/24 at 12:31**

## Pipelines

Total: **1**

- Successful: **0 (0%)**
- Partially successful: **1 (100%)**
- Unsupported: **0 (0%)**
- Failed: **0 (0%)**

### Job types

Supported: **1 (100%)**

- YAML: **1**

### Build steps

Total: **28**

Known: **27 (96%)**

- script: **10**
- bash: **9**
- PublishBuildArtifacts@1: **5**
- UsePythonVersion@0: **2**
- powershell: **1**

Unknown: **1 (3%)**

- CMake@1: **1**

Actions: **35**

- run: **20**
- actions/checkout@v4.1.0: **6**
- actions/upload-artifact@v4.1.0: **5**
- actions/setup-python@v5.0.0: **2**
- ./.github/actions/ci_conda_steps: **2**

### Triggers

Total: **2**

Known: **2 (100%)**

- pullRequest: **1**
- continuousIntegration: **1**

Actions: **2**

- pull_request: **1**
- push: **1**

### Environment

Total: **2**

Known: **2 (100%)**

- system_debug: **1**
- CPU_COUNT: **1**

Actions: **2**

- system_debug: **1**
- CPU_COUNT: **1**

### Other

Total: **20**

Known: **20 (100%)**

- matrix: **6**
- CIBW_TEST_REQUIRES: **2**
- CIBW_TEST_COMMAND: **2**
- CIBW_ENVIRONMENT: **2**
- CIBW_BUILD_VERBOSITY: **2**
- CIBW_BUILD: **2**
- CIBW_MANYLINUX_X86_64_IMAGE: **1**
- macOS_sdk_url: **1**
- macOS_sdk_filename: **1**
- maxParallel: **1**

Actions: **20**

- matrix: **6**
- CIBW_TEST_REQUIRES: **2**
- CIBW_TEST_COMMAND: **2**
- CIBW_ENVIRONMENT: **2**
- CIBW_BUILD_VERBOSITY: **2**
- CIBW_BUILD: **2**
- CIBW_MANYLINUX_X86_64_IMAGE: **1**
- macOS_sdk_url: **1**
- macOS_sdk_filename: **1**
- max_parallel: **1**

### Manual tasks

Total: **2**

Self hosted runners: **2**

- `macOS-latest`: **2**

### Partially successful

#### smurff/ExaScience.smurff

- [pipelines/smurff/ExaScience.smurff/.github/workflows/exascience.smurff.yml](pipelines/smurff/ExaScience.smurff/.github/workflows/exascience.smurff.yml)
- [pipelines/smurff/ExaScience.smurff/.github/actions/ci_conda_steps/action.yml](pipelines/smurff/ExaScience.smurff/.github/actions/ci_conda_steps/action.yml)
- [pipelines/smurff/ExaScience.smurff/config.json](pipelines/smurff/ExaScience.smurff/config.json)
- [pipelines/smurff/ExaScience.smurff/source.yml](pipelines/smurff/ExaScience.smurff/source.yml)
