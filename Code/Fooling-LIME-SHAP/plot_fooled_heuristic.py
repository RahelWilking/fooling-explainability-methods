import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy
import statistics

from matplotlib import pyplot as plt
import seaborn as sns

from setup_experiments import kl_divergence

#print(kl_divergence(np.array([1.0,0.0,0.0,0.0]),np.array([0.0,0.0,0.0,1.0])))
#print(kl_divergence(np.array([0.0,0.0,0.0,1.0]),np.array([1.0,0.0,0.0,0.0])))

# seed
#rng = np.random.default_rng(12345)

# draw random data
#data = rng.random(size=(100000, 4))

# normalize
#data = data/data.sum(axis=1, keepdims=True)

#scipy.special.softmax(data, axis=1)

distribution_one = []
distribution_two = []
step = 0.05
precision = 2 # an step anpassen
for val1 in np.arange(0, 1+step/2, step):
    for val2 in np.arange(0, 1+step/2-val1, step):
        for val3 in np.arange(0, 1+step/2-val1-val2, step):
            for secondval1 in np.arange(0, 1+step/2-val1, step):
                for secondval2 in np.arange(0, 1+step/2 - max(secondval1,val2), step):
                    for secondval3 in np.arange(0, 1+step/2 - max(secondval1+secondval2, val3) , step):
                        distribution_one.append([val1, val2, val3, round(1-val1-val2-val3, precision)])
                        distribution_two.append([secondval1, secondval2, secondval3, round(1-secondval1-secondval2-secondval3, precision)])

distribution_one = np.array(distribution_one)
distribution_two = np.array(distribution_two)

goal_dist_obscure = np.array([0, 0, 0, 1.0])
goal_dist_show = np.array([1.0, 0, 0, 0])

kl_dist_obscure = [kl_divergence(goal_dist_obscure, distribution_one[i]) for i in range(distribution_one.shape[0])]
kl_dist_show = [kl_divergence(goal_dist_show, distribution_two[i]) for i in range(distribution_two.shape[0])]

fooled_heuristic = [statistics.harmonic_mean([kl_dist_obscure[i], kl_dist_show[i]]) for i in range(len(kl_dist_show))]

sns.displot(fooled_heuristic, stat='percent', binwidth=0.25)
plt.ylabel("Prozent")
plt.xlabel("Fooled-Heuristik")

plt.savefig("fooled_heuristic_distribution.png", bbox_inches="tight")
