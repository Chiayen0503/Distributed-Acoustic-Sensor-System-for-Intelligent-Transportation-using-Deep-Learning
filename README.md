# Distributed Acoustic Sensor System for Intelligent Transportation using Deep Learning
We present DAS sample and 1D/2D CNN for vehicle type and occupancy classification

1. How to setup <br>
   Step 1: Download and unzip preprocessed data in current directory: https://drive.google.com/file/d/1Dqm0WuZqC7Iopgt7Yy12ddgDN6fU5WyW/view?usp=drive_link <br>

   or alternatively (not recommand)<br>

   Download and unzip raw data (1) in current directory and run the following notebooks (2):
   * (1) raw data: https://drive.google.com/file/d/1RvyaRBf5PyBU4nVys5bn6OjHhCfGOT1c/view?usp=drive_link
   * (2) generate 058 (5p) txt data.ipynb and generate 058.5 (5p_to_1p) txt data.ipynb

   Step 2: Install pandas, numpy, and tensorflow <br>
   Step 3: Keep in mind how the names of datasets in this repository are related to the paper:

   | name in paper                            | pre-proccessed data in the repository    | dataset description                 | labels description                                                                                                                        |
   |------------------------------------------|------------------------------------------|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
   | RC-60-Mix                                | 058_5p_to_1p_X.txt<br>058_5p_to_1p_y.txt | Car 2 with 5, 4, 3, 2, 1 passengers(p) | 5p: label=1<br>4p: label=2<br>3p: label=3<br>2p: label=4<br>1p: label=5                                                                   |
   | AllCars-1p <br>(RC-60-1p) | 026_X.txt<br>026_y.txt                   | Car 1, 2, 3, 4, 5 with 1 passenger  | Car1: label=1<br>Car2: label=2<br>Car3: label=3<br>Car4: label=4<br>Car5: label=5<br>(Note: we only retrieve car2 data from 026 data) |
   | RC-60-5p                                 | 058_5p_X.txt<br>058_5p_y.txt             | Car 2 with 5 passengers             | label=1                                                                                                                                   |

    

3. Current reproducing results:
   |                             | 5-way | 5-way | 2-way | 2-way | 2-way + | 2-way + |
   |-----------------------------|-------|-------|-------|-------|---------|---------|
   | Model of CNN                | 1D    | 2D    | 1D    | 2D    |   1D    |   2D    |
   | Train:Test                  | 67:33 | 80:20 | 80:20 | 80:20 |  80:20  |  80:20  |
   | RC-60-Mix                   | 0.81  | 0.93  | 0.896 | 0.98  |         |  0.96   |
   | Ind. AllCars-1p  (RC-60-1p) | 0.297 | 0.68  | 0.46  | 0.62  |         |  0.52   |
   | Ind. RC-60-5p               | 0.099 | 0.28  | 0.247 | 0.52  |         |  0.53   |
