# This source is referenced from from https://github.com/MrNeRF/MNIST_CUDA
# MNIST Classifier with CUDA and C++

This project is a MNIST classifier using CUDA and C++ to code an MLP from scratch. 
In its tests it uses the torch C++ API to assure correct implementation.
Attention: This not yet in a clean version, but it is working.

## Dependencies

- CMake (version >= 3.22)
- CUDA Toolkit (version >= 12.0) 
- PyTorch (libtorch)
- Google Test (release-1.10.0)
- Cutlass (version >= 3.1)

It might also work with a lower version of CUDA, but that is the only one I have tested.
CMake version might be also considerable lower. Just test it.
## Installation

Use `Google Colab` to run it.

```bash
git clone https://github.com/Hoangnhan1708/ANN_Parallel_Programming
cd ANN_Parallel_Programming
```
### libtorch

```bash
cd external
```

Download the `libtorch` library using the following command:

```bash
wget https://download.pytorch.org/libtorch/test/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip  
```

This will download a zip file named `libtorch-shared-with-deps-latest.zip`. To extract this zip file, use the command:

```bash
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip -d external/
rm libtorch-cxx11-abi-shared-with-deps-latest.zip
```

This will create a folder named `libtorch` in the `external` directory of your project.

### cutlass

Download the `cutlass` library using the following command:
```bash
git clone https://github.com/NVIDIA/cutlass
```
This will create a folder named `cutlass` in the `external` directory of your project.

### Fashion MNIST Data

```bash
cd ..
```

The Fashion MNIST data can be downloaded using the following command:

```bash
cd data
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
```

These commands will download four gzip files. To extract these gzip files, use the command:

```bash
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
cd ..
```


## Building the Project

To build the project, follow these steps:
1. Create a new directory named `build` and navigate into it:

    ```bash
    mkdir build && cd build
    ```

2. Run the CMake configuration:

    ```bash
    cmake -DCMAKE_BUILD_TYPE=Release ..
    ```
	If you have problems to configure the build, you might look up the graphics card architecture you are using.
	Then replace CUDA_ARCHITECTURE 89 with the number of your architecture.

3. Finally, compile the project:

    ```bash
    make -j$(nproc)
    cd ..
    ```

This will create an executable named `mnist_cuda` in the `build` directory.

Run it:
```bash
./build/mnist_cuda data
```

If you encounter an error, it may be because you have not granted execution permission.

```bash
!chmod +x ./build/mnist_cuda
```

## Running the Tests

After building the project, you can run the tests with the following command:

```bash
./build/cuda_kernel_tests
```
