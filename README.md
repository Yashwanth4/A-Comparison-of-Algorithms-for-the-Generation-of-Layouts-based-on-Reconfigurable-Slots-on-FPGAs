# FPGA Placement Optimization

**This project optimizes the placement of 2d reconfigurable slots on a Field-Programmable Gate Array (FPGA) using Simulated Annealing (SA) and Evolutionary Strategy (ES) algorithms.**

Structure:

* [ ] `  /Documents/Description     `  Description of the project

* [ ] `  /Documents/Presentations   `  Slides of the initial and final presentations

* [ ] `  /Documents/Thesis          `  Thesis file identical to the printed version

* [ ] `  /Documents/Thesis/src      `  TeX source files

* [ ] `  /Code                      `  The developed work including additional documentation

* [ ] `  /Templates		       `  Templates for Thesis, Presentations and the Declaration required in the Thesis

## Installation
To run this code, you'll need Python 3 and the following packages:
- numpy
- matplotlib

You can install them using pip:

````
pip install numpy matplotlib
````


## Usage
To run the code, navigate to the project directory and use the following command:


````
python <.pyfile> <file_name> <algorithm>
````

Replace <.pyfile> with the name of the py file you want to run. Here there are two py files that Optimization_SA.py and Optimization_ES.py and replace <file_name> with the name of the input file you want to use, and with the optimization algorithm you want to run. The available algorithms are "SA" for simulated annealing, "SAT" for simulated annealing with a constant time period, "SAB" for running the simulated annealing algorithm to achieve required bounding area and "SAWT" for running simulated annealing algorithm considering the Area Adjustment technique. 
For running ES algorithm replace 'SA' with 'ES' in the commands while choosing the algorithm.

The input file should be in the following format:

````
Outline: <width_of_FPGA> <height_of_FPGA>
<block_name1> <required area in microslots>
<block_name2> <required area in microslots>
<block_name3> <required area in microslots>
<block_name4> <required area in microslots>
````
Here's an example input file:

````
Outline: 16 16
Block1 28
Block2 20
Block3 27
Block4 1
````
## Output
The program outputs the insightful visualizations of the final placement of the blocks on the given FPGA. The output also provides the final bounding area and the runtime.

The sample output for the above given input file:

````
runtime = 34.18572378158569
final_bounding_area = 84
````

![failed](Code/Figures/sample_vis.png)

# FPGA Decision Problem
This project tries to tell you if the given reconfigurable slots can be placed on a given FPGA. Here in this work Zynq-7020 and Zynq-7030 has been considered.

## Installation
To run this code, you'll need Python 3 and the following packages:
- numpy 
- matplotlib
- sympy

You can install them using pip:
````commandline
pip install numpy matplotlib sympy
````

## Usage
To run the code, navigate to the project directory and use the following command:

````commandline
python <script.py> <file_name> <device_name>
````
Replace <script.py> with the name of the py file you want to run. Here there are two py files that Decision_SA.py and Decision_ES.py and replace <file_name> with the name of the input file you want to use, and with the device name. Here you can provide "Zynq7020" or "Zynq7030".

The input file should be in the following format for Zynq 7020:
````commandline
<block_name1> <required area in microslots>
<block_name2> <required area in microslots>
<block_name3> <required area in microslots>
<block_name4> <required area in microslots>
````
The dimensions of Zynq 7020 device is represented using 6 is width and 3 is height. pre-placed blocks are used to cover the regions of FPGA where there is no processing units.
For device Zynq 7030 the dimensions are 7 (width), 4 (height)
We have also created device files. These files contain fpga width, fpga height and required pre-placed blocks to represent the chosen device.
These files are loaded automatically once you input the device name in the command line.

Here's an example input file:
````commandline
Block1 3
Block2 2
Block3 3
Block4 2
````
# Output
The program outputs the insightful visualizations of the final placement of the blocks on the selected FPGA and outputs TRUE if all given reconfigurable slots are able to place on given FPGA or else FALSE.
The sample output for the above input file:
````
TRUE
````

![failed](Code/Figures/sample_dec.png)

Here in the Figure above the preplaced blocks are represented using black color blocks.

