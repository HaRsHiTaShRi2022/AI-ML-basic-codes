import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

total_test_subjects = len(dataset)
total_advertisement = 10
advertisement = []
ad_selected = [0]*total_advertisement
sum_of_rewards = [0]*total_advertisement
total_rewards = 0

for i in range(0, total_test_subjects):
    ad = 0
    max_upperbound = 0
    for j in range(0, total_advertisement):
        if ad_selected[j] > 0:
            average_reward = sum_of_rewards[j]/ad_selected[j]
            delta_i = math.sqrt((3*math.log(i+1))/(ad_selected[j]))
            upper_bound = average_reward+delta_i
        else:
            upper_bound = 1e404
        if max_upperbound < upper_bound:
            max_upperbound = upper_bound
            ad = j
    advertisement.append(ad)
    ad_selected[ad] += 1
    reward = dataset.values[i, ad]
    sum_of_rewards[ad] += reward
    total_rewards += reward

plt.hist(advertisement)
plt.xlabel('Ads')
plt.ylabel('Number of selections')
plt.title('Histogram of ads selections')
plt.show()
