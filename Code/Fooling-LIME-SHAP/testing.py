from utils import *
from get_data import *

params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_german(params)

features = [c for c in X]
X["y"] = y

print(X.corr()["Gender"])