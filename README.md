# Installation
Steps for installing on Linux:
* Create Python environment: `conda create -n social_rearrange -y python=3.8 cmake=3.14.0` then activate the conda environment.
* Install the open source Habitat Sim platform: `conda install -y habitat-sim withbullet  headless -c conda-forge -c aihabitat-nightly`
* Install the open source Habitat Lab platform: 
    * `git clone https://github.com//habitat-lab.git`
    * `cd habitat-lab`
    * `pip install -e habitat-lab`
* Install additional requirements: 
    * `pip install -r bdp/rl/requirements.txt`
    * `pip install -r bdp/rl/ddppo/requirements.txt`
    * `pip install -e .`
* Download datasets: `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`

# Run Commands
Commands for running our proposed BDP method and baselines are below. To run with state instead of visual input, replace `im.yaml` with `st.yaml`. To run with different tasks, add command line arguments `TASK_CONFIG.DATASET.DATA_PATH data/datasets/replica_cad/rearrange/v1/train/tidy_house_10k_1k.json.gz TASK_CONFIG.TASK.TASK_SPEC tidy_house_multi` where `tidy_house` can be replaced with `prepare_groceries` or `set_table`.
* BDP: `python bdp/run.py --exp-config bdp/config/hab/im.yaml --run-type train WRITER_TYPE wb RL.AGENT_SAMPLER.TYPE "PrefPlayAgentSampler" RL.POLICIES.POLICY_0.batch_dup 2 RL.PPO.num_mini_batch 4 RL.POLICIES.POLICY_0.high_level_policy.PREF_DIM 8 RL.AGENT_SAMPLER.PREF_DIM 8 TOTAL_NUM_STEPS 2e8 RL.AGENT_SAMPLER.SECOND_STAGE_START 1e8 RL.AGENT_SAMPLER.REUSE_VISUAL_ENCODER True RL.POLICIES.POLICY_0.high_level_policy.use_pref_discrim True`
* PBT: `python bdp/run.py --exp-config bdp/config/hab/im.yaml --run-type train WRITER_TYPE wb RL.AGENT_SAMPLER.NUM_AGENTS 9 RL.AGENT_SAMPLER.SECOND_STAGE_START 1e8 TOTAL_NUM_STEPS 2e8`
* SP: `python bdp/run.py --exp-config bdp/config/hab/im.yaml --run-type train WRITER_TYPE wb RL.AGENT_SAMPLER.NUM_AGENTS 9 RL.AGENT_SAMPLER.SECOND_STAGE_START 1e8 TOTAL_NUM_STEPS 2e8 RL.AGENT_SAMPLER.ONLY_SELF_SAMPLE True RL.AGENT_TRACKER.RENDER_SELF True RL.AGENT_SAMPLER.SELF_PLAY True`
* TrajeDi: `python bdp/run.py --exp-config bdp/config/hab/im.yaml --run-type train WRITER_TYPE wb RL.AGENT_SAMPLER.NUM_AGENTS 9 RL.AGENT_SAMPLER.SECOND_STAGE_START 1e8 TOTAL_NUM_STEPS 2e8 RL.POLICIES.POLICY_0.high_level_policy.div_reward True`
* FCP (replace `LOAD_POP_CKPT` argument with the path to a pre-trained population checkpoint): `python bdp/run.py --exp-config bdp/config/hab/im.yaml --run-type train WRITER_TYPE wb RL.AGENT_SAMPLER.NUM_AGENTS 9 RL.AGENT_SAMPLER.SECOND_STAGE_START 0.0 TOTAL_NUM_STEPS 2e8 RL.AGENT_SAMPLER.LOAD_POP_CKPT im_pp8_st_I88d77ccb RL.AGENT_SAMPLER.FORCE_CPU True`

