# Distributed Acoustic Sensor System for Intelligent Transportation using Deep Learning
We present DAS sample and 1D/2D CNN for vehicle type and occupancy classification

1. How to setup <br>
   Step 1: Download and unzip preprocessed data in current directory: https://drive.google.com/file/d/1Dqm0WuZqC7Iopgt7Yy12ddgDN6fU5WyW/view?usp=drive_link <br>

   or alternatively <br>

   Download and unzip raw data (1) in current directory and run the following notebooks (2):
   * (1) raw data: https://drive.google.com/file/d/1RvyaRBf5PyBU4nVys5bn6OjHhCfGOT1c/view?usp=drive_link
   * (2) generate 058 (5p) txt data.ipynb and generate 058.5 (5p_to_1p) txt data.ipynb

   Step 2: Install pandas, numpy, and tensorflow 

2. Current reproducing results:
   |                             | 5-way | 5-way | 2-way | 2-way | 2-way + |
   |-----------------------------|-------|-------|-------|-------|---------|
   | Model of CNN                | 1D    | 2D    | 1D    | 2D    |         |
   | Train:Test                  | 67:33 | 80:20 | 80:20 | 80:20 |         |
   | RC-60-Mix                   | 0.81  | 0.93  | 0.896 | 0.98  |         |
   | Ind. RC-60-1p  (AllCars-1p) | 0.297 | 0.68  | 0.46  | 0.62  |         |
   | Ind. RC-60-5p               | 0.099 | 0.28  | 0.247 | 0.52  |         |
