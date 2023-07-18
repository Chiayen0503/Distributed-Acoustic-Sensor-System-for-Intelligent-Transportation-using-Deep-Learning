# Distributed Acoustic Sensor System for Intelligent Transportation using Deep Learning
We present DAS sample and 1D/2D CNN for vehicle type and occupancy classification

1. **How to setup <br>**
   Step 1: Download and unzip preprocessed data in current directory: https://drive.google.com/file/d/1ore1g5sN8bUA7NvG5lvAGw9z6_LQlRuO/view?usp=drive_link <br>

   or alternatively (not recommand)<br>

   Download and unzip raw data (1) in current directory and run the following notebooks (2):
   * (1) raw data: https://drive.google.com/file/d/1RvyaRBf5PyBU4nVys5bn6OjHhCfGOT1c/view?usp=drive_link
   * (2) generate 058 (5p) txt data.ipynb and generate 058.5 (5p_to_1p) txt data.ipynb

   Step 2: Install pandas, numpy, and tensorflow <br>
   Step 3: Keep in mind how the names of datasets in this repository are related to the paper:
   | name in paper                            | pre-proccessed data in the repository    | dataset description                                                                                                                             |
   |------------------------------------------|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
   | RC-60-Mix                                | 058_5p_to_1p_X.txt<br>058_5p_to_1p_y.txt | Car 2 with 5, 4, 3, 2, 1 passengers                                                                                                             |
   | AllCars-1p <br>(= is now re-named to RC-60-1p) | 026_X.txt<br>026_y.txt                   | The file contains signals from <br>Car 1, 2, 3, 4, 5 with 1 passengers (a driver)<br>but only Car 2 data is used for this individual testing dataset   |
   | RC-60-5p                                 | 058_5p_X.txt<br>058_5p_y.txt             | Car 2 with 5 passengers                                                                                                                         |

    

2. **How to use this repository <br>**

   We try to reproduce the testing results from Table III and Table IIV in our paper (named the same as this repository) so we named jupyter notebooks the same as the experiments in Table III and Table IIV. 

   For example, in ```5 way - 1d.ipynb```,  we trained a 1dcnn to classifiy exact number of passengers (5 way: 5 classes each of them has different number of passenger from 1 to 5).




3. **Current reproducing results:<br>**
   |                             | 5-way | 5-way | 2-way | 2-way | 2-way + | 2-way + |
   |-----------------------------|-------|-------|-------|-------|---------|---------|
   | Model of CNN                | 1D    | 2D    | 1D    | 2D    |   1D    |   2D    |
   | Train:Test                  | 67:33 | 80:20 | 80:20 | 80:20 |  80:20  |  80:20  |
   | RC-60-Mix                   | 0.81  | 0.93  | 0.896 | 0.98  |         |  0.96   |
   | Ind. AllCars-1p  (RC-60-1p) | 0.297 | 0.68  | 0.46  | 0.62  |         |  0.52   |
   | Ind. RC-60-5p               | 0.099 | 0.28  | 0.247 | 0.52  |         |  0.53   |
