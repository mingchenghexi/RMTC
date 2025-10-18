# RMTC

Set the `SUMO_HOME` environment variable to point to your SUMO installation within your Python environment:

```bash
export SUMO_HOME=/your_python_env_path/lib/python3.6/site-packages/sumo
```

Unzip the required traffic scenario data:

```bash
cd onpolicy/envs/sumo_files_marl/scenarios
unzip fenglin.zip
cd ../../../
```

Run the training script to start RMTC training:

```bash
python onpolicy/scripts/train/train_sumo.py
```

