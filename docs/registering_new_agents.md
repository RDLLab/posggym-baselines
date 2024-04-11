# Registering RL PyTorch agents to POSGGym

Given an PyTorch RL agent that has been saved to a checkpoint, you can register it with posggym by following the steps below:

## 1. Save the new policy models state

We need to save the policy configuration and it's weights to a file. This can be done using the `save_to_posggym_format.py` script.:

For example, given `checkpoint*.pt` checkpoints stored in `my_checkpoint_dir`, you can run:

```bash
python save_to_posggym_format.py \
    --checkpoint_dir ~/path/to/my_checkpoint_dir \
    --output_dir ~/path/to/output_dir
```

This will load the extract the latest checkpoint weights, reformat into the format that the `posggym.agent.TorchPolicy` expects, and save this as in pickle format in `output_dir`.

## 2. Save the new policy file into the `posggym-agent-models` repository

The `posggym-agent-models` repository contains the model weights for the `posggym.agents`. You need to add the policy you want to register to this repository, being careful to add it along the file path that will match exactly where the policy will be registered within `posggym.agents`.

For example, to add a policy for the `DrivingContinuous-v0` environment, where the policy will be registered from the file `posggym.agents.continuous.driving.__init__.py`. The policy file should be added to the `posggym-agent-models` repository at `posggym-agent-models/posggym/agents/continuous/driving/` directory or a subdirectory of this directory.

For an example you can look at `posggym.agents.grid_world.driving.__init__.py` and the corresponding policy files in `posggym-agent-models/posggym/agents/grid_world/driving/`.

**Note** June 16, 2023: the naming of directories has changed a bit, so see the `posggym-agent-models` repo for the correct directory structure/naming conventions.

## 3. Register the new policy

Once the policy file has been added to the `posggym-agent-models` repository, you can register it with posggym by adding a file which loads it's spec to `posggym.agents`.

For example, to register the policy `sp_seed0.pkl` for the `DrivingContinuous-v0` environment, you would add registration code to `posggym.agents.continuous.driving.__init__.py`. An example of this can be seen in `posggym.agents.grid_world.driving.__init__.py`.

Make sure you add the correct observation and action processors, since the policy needs to be able to work with the unwrapped environment, compared to the wrapped environment used during training with rllib.

Make sure also to update `posggym.agents.__init__.py` to register the new policy's spec.

## 4. Test policy

You can then test your policy is registered correctly, by copying the `model_state.pkl` (i.e. `sp_seed0.pkl`) file into the correct directory in `posggym/assets/agents/` and then try to load the policy with `posggym.agents.make` (you may need to use `posggym.agents.pprint_registry` to figure out what the ID for the policy is.)

## 5. Update `posggym` docs

Update the `posggym.agents` docs.

- Run `posggym/docs/scripts/gen_agent_mds.py` to generate/update environment agent's docs page to include the new agents.

If the agent is for a new environment then you may also need to update some other doc pages to include the new environment in their toc tree.

## 6. Push the changes to the `posggym-agent-models` repository

If all is working correctly, then push the new `model_state.pkl` to the `posggym-agent-models` repo.

To test all is working properly, you can remove the `model_state.pkl` file from `posggym/assets/agents/` and try to load the policy again. If it downloads and then loads correctly, then it is being loaded from the `posggym-agent-models` repo.

## 7. Push changes to `posggym-agents` repository

Finally, push the changes to the `posggym.agents` repository.
