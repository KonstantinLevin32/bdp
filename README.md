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

# Directory Structure
* `bdp/`: Contains all the task code, trainer code, and configuration files.
    * `task/`: The sensor and measurement definitions for the Social Rearrangement task.
    * `rl/`: All trainer code.
        * `ddppo/`: Distributed PPO updater.
        * `hrl/`: Code for the HRL policy and skills.
        * `multi_agent/`: Multi-agent sampling strategies.
        * `ppo_trainer.py`: Main training loop.
    * `config/`: YAML configuration files.
        * `hab/`: Config files for different approaches.
        * `hc_agents/`: Config files defining behavior of hard coded ZSC evaluation agents.
        * `tasks`: Task definition config files for Social Rearrangement tasks.
    * `utils/`: Shared modules for policies and visualization utilities.
    * `common/`: Shared modules for the RL trainer.

# Run Commands
Commands for running our proposed BDP method and baselines are below. 
* BDP: `python bdp/run.py --exp-config bdp/config/hab/bdp.yaml --run-type train`
* PBT: `python bdp/run.py --exp-config bdp/config/hab/pbt.yaml --run-type train`
* SP: `python bdp/run.py --exp-config bdp/config/hab/sp.yaml --run-type train`
* TrajeDi: `python bdp/run.py --exp-config bdp/config/hab/im.yaml --run-type train`
* FCP `python bdp/run.py --exp-config bdp/config/hab/fcp.yaml --run-type train RL.AGENT_SAMPLER.LOAD_POP_CKPT pretrained_pop.ckpt` Replace the `LOAD_POP_CKPT` argument with the path to a pre-trained population checkpoint `.ckpt` file.

To run with different tasks, add command line arguments `TASK_CONFIG.DATASET.DATA_PATH data/datasets/replica_cad/rearrange/v1/train/tidy_house_10k_1k.json.gz TASK_CONFIG.TASK.TASK_SPEC tidy_house_multi` where `tidy_house` can be replaced with `prepare_groceries` or `set_table`.

