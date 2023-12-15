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

## Details
