import subprocess
from sklearn.model_selection import ParameterGrid
from scipy.stats import loguniform
import numpy as np


if __name__ == "__main__":
    encoder_name = "efficientnet_b0"
    seed = 42
    lower_lr = 1e-4
    upper_lr = 1e-3
    bs = 512
    np.random.seed(seed)
    learning_rates = loguniform.rvs(lower_lr, upper_lr, size=5)
    param_grid = {'learning_rate': learning_rates}
    parameters = list(ParameterGrid(param_grid))
    for idx, val in enumerate(parameters):
        print(f"starting run {idx+1}/{len(parameters)} with encoder: {encoder_name} and lr: {val['learning_rate']}")
        subprocess.run(f"python3 train.py --run_number {idx+1} --encoder {encoder_name} --lr {val['learning_rate']} --batch_size {bs}", shell=True, check=True)
