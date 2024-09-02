import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

total_test_subjects = len(dataset)
total_ads = 10
advertisement = []
reward_one = [0]*total_ads
reward_zero = [0]*total_ads
total_reward = 0
for i in range(0, total_test_subjects):
    ad = 0
    max_random = 0
    for j in range(0, total_ads):
        random_beta = random.betavariate(reward_one[j]+1, reward_zero[j]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = 1
    advertisement.append(ad)
    reward = dataset.values[i, ad]
    if reward == 1:
        reward_one[ad] = reward_one[ad] + 1
    else:
        reward_zero[ad] = reward_zero[ad] + 1
    total_reward = total_reward + reward

plt.hist(advertisement)
plt.show()


