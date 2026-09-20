import numpy as np
import os
for num_sensors in [150, 200, 250, 300, 350, 400, 600, 1000, 2000]:
    if not os.path.exists(f'{num_sensors}_sensors'):
        os.mkdir(f'{num_sensors}_sensors')
    os.chdir(f'{num_sensors}_sensors')
    for data_set in range (1, 21):
        with open(f'{num_sensors}_{data_set}.txt', 'w+') as f:
            f.write(str(num_sensors) + '\n')
            for i in range (num_sensors):
                width = min(np.random.exponential(1/0.005), 2000)
                length = min(np.random.exponential(1/0.005), 2000)
                depth = np.random.uniform(0, 1500)
                f.write(f'{width} {length} {depth}\n')
    os.chdir('..')
