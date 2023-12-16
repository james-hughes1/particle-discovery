# Principles of Data Science Coursework Repository

## Description
This repository contains code that is used to statistically investigate an experiment in which we are interested in discovering signal events from a sample of events with associated random variable $M$.

The functionality enables users to sample from a mixed distribution where the background events are modelled as having an exponential distribution in $M$, whereas the value of $M$ for the signal events is a narrow spike around some critical value, modelled as a Gaussian distribution. Further functions enable users to estimate the sample size required to achieve a discovery rate of 90% at some specified signficance level.

## How to use the project
The project is designed to be usable via Docker. In order to run the project in this way, it is best to clone the repository to your local device using git, build the docker
image using the terminal command

`docker build -t s1_jh2284 .`

and then running the container by executing the command

`docker run --rm -ti s1_jh2284`

where the options specify that the container is removed after exiting, and ensure that the container can be interacted with easily in the terminal.
Alternatively, you can run (in the root directory)

`conda env create -f environment.yml`

`conda activate mphildis_assessment_jh2284`

to build the correct conda environment and run the code in the location you cloned it to.
Next, to run the actual code you simply execute the command

`python src/solve_part_[c-g].py`

The codebase is highly modular in order to make the individual scripts more readable, and to have less duplicated code. The scripts for parts c and d use routines created in `mixed_pdf_tools`, while the scripts for the rest of the questions use more advanced routines in `simulation_tools`. In particular, this module initialises the global variables used for the inverse CDF sampling whenever they are imported, so you may notice a brief delay of around 30 seconds when running the scripts for parts e-g.

## Computer specification details

The repository was developed on a system with specifications
 - Processor: Intel(R) Core(TM) i5-1035G1 CPU @ 1.00GHz, 1190 Mhz, 4 Core(s), 8 Logical Processor(s)
 - Installed Physical Memory (RAM): 8.00 GB

The following gives approximate execution times using this system:
 - `python src/solve_part_c.py` 5 secs
 - `python src/solve_part_d.py` 5 secs
 - `python src/solve_part_e.py` 30 secs
 - `python src/solve_part_f.py` 45 mins
 - `python src/solve_part_g.py` 2 hrs
 - `docker build -t s1_jh2284 .` 5 mins
