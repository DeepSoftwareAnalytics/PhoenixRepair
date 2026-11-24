# Competitive runs

!!! abstract "Running swe-agent competitively on benchmarks"
    This page contains information on our competitive runs on SWE-bench, as well as tips and tricks for evaluating on large batches.

    * Please make sure you're familiar with [the command line basics](cl_tutorial.md) and the [batch mode](batch_mode.md)
    * The default examples will be executing code in a Docker sandbox, so make sure you have docker installed ([docker troubleshooting](../installation/tips.md)).


## Current competitive configurations

!!! hint "Most recent configs"
    You can find all benchmark submission configs [here](https://github.com/SWE-agent/SWE-agent/tree/main/config/benchmarks)

Examples of configurations for SWE-bench submissions:

* [250225_anthropic_filemap_simple_review.yaml](https://github.com/SWE-agent/SWE-agent/blob/main/config/250225_anthropic_filemap_simple_review.yaml):
  This is our current default one-attempt config. It uses `claude-3-7-sonnet-20250219`.
* [250212_sweagent_heavy_sbl.yaml](https://github.com/SWE-agent/SWE-agent/blob/main/config/250212_sweagent_heavy_sbl.yaml):
  This config runs 5 attempts with slightly different configurations using `claude-3-7-sonnet-latest`,
  then uses o1 to discriminate between them.
  This is a very expensive configuration.
  If you use it, also make sure to use Claude 3.7 instead of claude 3.5.

!!! warning "Retry configurations and command line arguments"
    Note that the structure of the configuration with agents that run multiple attempts is different from the one of the
    default agent. In particular, supplying options like `--agent.model.name` etc. will cause (potentially confusing)
    error messages. Take a look at the above configuration file to see the structure!

You can find the command with which to run each config at the top of the config file.

In order to run on multiple workers with Claude, you need to use multiple API keys in order to have enough cache break points.
For this, please set the following environment variable before running

```bash
# concatenate your keys
export CLAUDE_API_KEY_ROTATION="KEY1:::KEY2:::KEY3"
```

See our [notes on Claude](../config/models.md) for more details.

## Memory consumption

We run our configuration on a machine with 32GB memory and 8 cores.
To avoid out-of-memory (OOM) situations, we recommend setting

```bash
--instances.deployment.docker_args=--memory=10g
```

limiting the maximum amount of memory per worker.

In our case, this completely avoided any instances of running OOM.

## Tradeoffs between resolution rate and cost

* Running multi-attempt configurations will always be _very_ expensive. Don't use them if cost is of importance.
* The simplest setting to keep cost in check is the per instance cost limit or turn limit.
  Without limiting cost, the average cost will also converge to infinity, as the agent will never stop iterating.
  With Claude 3.7, a cost-conservative limit would be $1 instance limit or lower and a turn count limit of 50.
  For our swe-bench submissions we use slightly higher limits (see the configs above).
