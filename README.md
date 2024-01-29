# Parallel programming semestral assignment

## Intro
Welcome to my parrallel programming course semestral assignment solution. 
This source code represents a program that finds a correlation formula between two vectors of values.

In this work, concepts such as vectorization and padding, OpenCL a GPGPU computation and genetic programming were used.

## Prerequisites
- CMake version >= 3.13
- C/C++ compiler (GCC) with C++17 support
- For building out from source - OpenCL SDK and OpenCL C++ headers in PATH (refer to the build guide below), Intel TBB (due to std::execution_policy)

## Run application
For Windows and Mac (**Apple Silicon only**), binaries are included in the **bin** directory.
For others, please refer to the build guide.

1. Download the resources files, available [HERE](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/).
2. Inside project folder, create a new folder called *"resources"* and inside, place the downloaded contents. The structure of the resources folder **must** be following:

    > resources/XXX/ACC_XXX.csv\
    > resources/XXX/HR_XXX.csv

    Where XXX represents patient number between 001 and 016
3. **IMPORTANT!** Replace the original HR_001.csv with the provided HR_001.csv in the root of the repository (due to wrong format inside the original dataset). The data have **NOT** been tampered with.
    > The regular expressions used for reformatting the HR source file are available in the *regex.txt* file
4. Open a terminal or a command line in the **root** directory of this project and run these using the following commands:
    ```bash
        ./build/exec/ppr <opt: period_size>
    ```

## Build from source
Build from source is supported **only** on Mac/Linux (Windows not tested).
1. Clone the repository.
2. Download the resources files, available [HERE](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/).
3. Inside the cloned repository, create a new folder called *"resources"* and inside, place the downloaded contents. The structure of the resources folder **must** be following:

    > resources/XXX/ACC_XXX.csv\
    > resources/XXX/HR_XXX.csv

    Where XXX represents patient number between 001 and 016
4. **IMPORTANT!** Replace the original HR_001.csv with the provided HR_001.csv in the root of the repository (due to wrong format inside the original dataset). The data have **NOT** been tampered with.
    > The regular expressions used for reformatting the HR source file are available in the *regex.txt* file

5. If you're building on Windows, make sure to set proper compiler path inside the **CMakeLists.txt** file.
6. Make sure you have the following environment variables set (based on your compiler), as shown in the following example:
    > CC="gcc-13" \
    CXX="g++\-13" \
    Optionally, you can adjust these inside the CMakeLists.txt file (CMAKE_C_COMPILER and CMAKE_CXX_COMPILER)
7.  Also make sure you have OpenCL and Intel TBB libraries installed properly
8. Build out the application using the **build.(sh|bat)** script (make sure it is executable)
9. Built binary will be available inside the **build/exec** directory. To run the binary, navigate back to the root of the project, open a command line or a terminal inside and type in the following command:
    ```bash
        ./build/exec/ppr <opt: period_size>
    ```

10. All of the logs will be placed inside the *log* folder
11. All of the generated outputs (SVG plots) will be placed inside the *out* folder
