# pytorch examples
Basic torch scripts which require at least one GPU.

## Run locally
```shell script
conda activate TestEnv
python example1.py
python example2.py
```

## Run in slurm

```shell script
export conda_bin=/home/kaiser/anaconda3/tmp/bin/activate   # REPLACE WITH YOUR OWN CONDA PATH

sbatch  --export=CONDA_BIN=$conda_bin --partition=tnt slurm_execution.sh 
```