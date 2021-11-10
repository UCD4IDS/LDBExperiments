# Feature Extraction for Signal Classification using Local Discriminant Basis
Tutorials and experiments using **Local Discriminant Basis (LDB)**.
In this Pluto notebook we describe the algorithm behind LDB and test its classification capabilities by applying it on two standard signal classification datasets.

## Authors
This note book is authored by Shozen Dan and Zeng Fung Liew under the supervision of Professor Naoki Saito at the University of California, Davis.

## Table of Contents
1. [Setup](#setup)
2. [Results](#results)
3. [Pluto notebook containing results and report](notebooks/LDBexperiment.jl)

## How to Open and Run Pluto Notebook <a name="setup"></a>
### Method 1 (**Recommended**): Opening notebook by cloning this repository
1. Clone the repository by typing the following:
```shell
git clone https://github.com/UCD4IDS/LDBExperiment.git
```
2. Navigate to the LDBExperiment directory and open up the Julia REPL.
3. Ensure Julia is working on the current directory. This can be checked using the following commands:
```julia
# shows the current working directory
julia> pwd() 

# change to the directory containing all the files from this repository. Eg:
# Windowns
julia> cd("C:/Users/USER/Documents/LDBExperiments")

# Linux, Mac
julia> cd("~/Documents/LDBExperiments")
```
4. Enter the package manager in the REPL by typing `]`. The following should be observed:
```julia
(@v1.6) pkg> 
```
5. Activate the current environment by typing the following.   
Note: Steps 3-4 has to be done correctly for this step to work!
```julia
(@v1.6) pkg> activate ./notebooks
(@v1.6) pkg> instantiate
```  

6. Exit the package manager mode by hitting the backspace key. Then, type in the following commands:
```julia
julia> import Pluto; Pluto.run()
```

7. Pluto should open up in the default browser. Open up the file by keying in the file path.

### Method 2: Opening notebook directly without downloading any files from this repository
1. Open up the Julia REPL.
2. Manually install the required packages for running the notebooks. The list of required packages can be found in the [Project.toml](notebooks/Project.toml) file under the notebook directory.  
Install the packages in Julia using either the REPL or through the package manager. The package manager can be activated by hitting the `]` key. Example:
```julia
# install on REPL
julia> using Pkg; Pkg.add("Pluto")
# install on package manager
(@v1.6) pkg> add Pluto
```
3. Return to the REPL and type the command below. If you are currently at the package manager mode, you can return to the REPL by hitting the backspace key.
```julia
julia> import Pluto; Pluto.run()
```
4. Pluto should open up in the default browser. Copy-paste the following URL into the file path:  
[https://github.com/ShozenD/LDBExperiments/blob/master/notebooks/LDBexperiment.jl](https://github.com/ShozenD/blob/master/notebooks/LDBexperiment.jl)

**Note:** When opening the notebooks using this method, Julia automatically downloads the notebook into the `~/.julia/pluto_notebooks` folder in your local machine. You may want to delete them once you are done.
