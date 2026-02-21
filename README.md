# PhoenixRepair: Rising from Multi-Location Sampling and Iterative Self-Reflection

> **PhoenixRepair** is a multi-agent framework that systematically explores multiple candidate location sets and performs iterative reflection and refinement on patch generation, thereby expanding the search space of repair strategies for automated issue resolution.

## 🔥 Highlights

- **State-of-the-art Performance**: Achieves **74.4% pass@1** on SWE-bench Verified with DeepSeek-V3.2
- **Superior Fault Localization**: Demonstrates higher fault localization accuracy across file, module, and function granularities
- **Model-Agnostic**: Consistent performance gains across different LLMs (DeepSeek-V3.1, DeepSeek-V3.2, Qwen-Coder-Plus)
- **Generalizable Framework**: Successfully transfers to other agent-based methods (Mini-SWE-agent, Live-SWE-agent)

## 📊 Performance

### Pass@1 on SWE-bench Verified

| Method | LLM | Resolved (%) |
|--------|-----|--------------|
| Mini-SWE-agent | DeepSeek-V3.2 | 61.0% |
| Live-SWE-agent | DeepSeek-V3.2 | 63.8% |
| SWE-Search | DeepSeek-V3.2 | 65.4% |
| Agentless | DeepSeek-V3.2 | 67.0% |
| Trae-agent | DeepSeek-V3.2 | 67.8% |
| SWE-agent | DeepSeek-V3.2 | 69.4% |
| Moatless-tools | Claude-4-Sonnet | 70.8% |
| **PhoenixRepair** | **DeepSeek-V3.2** | **74.4%** (+5.0%) |

### Fault Localization Accuracy

| Method | LLM | File Acc@1 | Module Acc@1 | Function Acc@1 |
|--------|-----|------------|--------------|----------------|
| SWE-agent | DeepSeek-V3.2 | 83.04% | 74.22% | 63.42% |
| **PhoenixRepair** | **DeepSeek-V3.2** | **85.64%** (+2.60%) | **76.82%** (+2.60%) | **66.64%** (+3.22%) |

## 🏗️ Architecture

PhoenixRepair comprises three main phases:

### 1️⃣ Multi-Location Sets Sampling

- **Localization Sampling Agent**: Iteratively samples N diverse location sets
- **Graph-based Localization** (for difficult tasks): Provides cross-file dependency information
- **Deduplication**: Removes duplicate location sets to obtain final candidates

### 2️⃣ Iterative Reflection & Refinement

- **Coder Agent**: Generates patches constrained to specific location sets
- **Analysis & Test Agent**: Evaluates patch quality through:
  - Test quality assessment
  - Regression test pass rate
- **Selector Agent**: Selects top-performing patches
- **Analysis Agent**: Distills guidance from historical attempts
- **Iterative Refinement**: Continues until converging to final location set

### 3️⃣ Final Round Generation

- Analysis agent distills insights from all historical attempts
- Generates final patch guided by comprehensive distilled knowledge


## 🚀 Quick Start

### Configuration

```bash
# Set up your API keys
export OPENAI_API_KEY="your-api-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
```

### Running PhoenixRepair

#### Phase 1: Multi-Location Sets Sampling
```bash
sweagent run-locate \
    --config config/locate.yaml \
    --agent.model.name "" \  # Please enter the model name, for example "deepseek-chat"
    --agent.model.api_base "" \  # Please enter the url, for example "https://api.deepseek.com/v1"
    --agent.model.per_instance_cost_limit 3.00 \
    --instances.type swe_bench \
    --instances.subset verified \
    --instances.split test \
    --enable_multi_sampling=True \
    --num_samples=5 \   # The Number of sequential samples per task
    --num_workers=25 \  # The number of tasks executed in parallel
    --enable_best_sample_selection=False \
    --comparison_model_name="" \  # Please enter the model name, for example "deepseek-chat"
    --comparison_api_key="" \  # Please enter the api_key
    --comparison_api_base="" \  # Please enter the url, for example "https://api.deepseek.com/v1"
```

#### Phase 2: Iterative Reflection & Refinement

```bash
sweagent run-batch \
    --config config/default.yaml \ 
    --agent.model.name "" \   # Please enter the model name, for example "deepseek-chat"
    --agent.model.api_base "" \  # Please enter the url, for example "https://api.deepseek.com/v1"
    --agent.model.per_instance_cost_limit 3.00 \
    --instances.type swe_bench \
    --instances.subset verified \
    --instances.split test \
    --enable_multi_sampling=True \
    --num_samples=3 \
    --comparison_model_name="" \  # Please enter the model name, for example "deepseek-chat"
    --comparison_api_key="" \   # Please enter the api_key
    --comparison_api_base="" \  # Please enter the url, for example "https://api.deepseek.com/v1"
    --deduplicated_patches_root ""  # Please enter the absolute path obtained in Phase 1， for example "/home/PhoenixPepair/trajectories/root/locate__openai--GLM-4.7__t-0.70__p-1.00__c-0.00___swe_bench_verified_test"
```






