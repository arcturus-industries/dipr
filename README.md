Deep Inertial Prediction
----------------------------------
*For more information and context related to this repo, please refer to our [website](https://dipr.ai).*

## Getting Started (non Docker)
Note: you will need to have pytorch installed (tested with 1.8 and higher)
```bash
python3 -m venv <venv_path>
source <venv_path>/bin/activate

git clone https://github.com/arcturus-industries/dipr.git && cd dipr
pip3 install -e .
python3 dipr/evaluate.py --challenge_folder <data_path>
```

## Getting Started (with Docker)
You will need `docker` and `realpath` commands to be installed
```
git clone https://github.com/arcturus-industries/dipr.git && cd dipr
# on x86_64 systems
./build-and-run.sh <data_path>
# on arm64 systems (like mac M1)
./build-and-run-aarch64.sh <data_path>
```
M1 Mac note: You can use either the X86_64 container or the arm64 container.  If you use the x86_64 container, you may see "Could not initialize NNPACK! Reason: Unsupported hardware." This is only a warning. It will however take a long time to run (about 30 minutes or longer after the docker build finishes)

## Package Content

 - `dataset.py` - sample API to read the challenge hdf5 dataset format
 - `cnn_backend.py` - a file with CNN inference backends (currenly only TorchScript is supported). If you plan to work on a DL inference framework other than TorchScript, implement it there
 - `noise_utils.py` - a file with noise calibration and parameters, you may adjust them to generate your own noise levels
 - `imu_fallback.py`  - a sample implmentation of ImuFallback with CNN velocity measurements
 - `utils.py` - auxiliary tools
 - `evaluate.py` - sample test script that runs ImuFallback on available datasets and outputs Mean Absolute Velocity metric


## Running sample evaluation script


```bash
python3 evaluate.py --challenge_folder <data_path>
```

or for the docker versions
```bash
# on x86_64 systems
./build-and-run.sh <data_path>
# on arm64 systems (like mac M1)
./build-and-run-aarch64.sh <data_path>
```

It will output something like:
```
python3.9 evaluate.py -df shared
Dataset OpenVR_2021-09-02_17-40-34-synthetic, segments durations [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0 ] sec
Processing datasets: 100%|██████████| 1/1 [05:04<00:00, 304.92s/files]
all_vel_mae_cnn 2.12cm/s
all_vel_mae_fallback 9.73cm/s
all_pose_mae_fallback 15.51cm
```

Which mean it found `OpenVR_2021-09-02_17-40-34-synthetic` test dataset, and executed ImuFallback on 13 segments of duration 7 seconds, and estimated over them averaged Mean Absolute Velocity Error as 9.73cm/s

It also outputs image with tracking plots to `<challenge_folder_root>/_results/<datasetname>.png`. There are plots for IMU only tracking, ImuFallback + CNN traking and ground truth

## Challenge folder Content

`train_synthetic` - a folder with train datasets, available after sign-up https://dipr.ai/sign-up

`test_synthetic` - a folder where evaluation script looks for test datasets (we share only one example dataset)

`_results` - a folder where evaluation script stores some results

`pretrained` - an example CNN model we ship

## Known Issues

Installing **dependencies** natively on Apple Silicon may fail with the following:
```bash
> pip3 install -e .
...
    error: Command "clang -Wno-unused-result -Wsign-compare -Wunreachable-code -fno-common -dynamic -DNDEBUG -g -fwrapv -O3 -Wall -iwithsysroot/System/Library/Frameworks/System.framework/PrivateHeaders -iwithsysroot/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.8/Headers -arch arm64 -arch x86_64 -Werror=implicit-function-declaration -ftrapping-math -DNPY_INTERNAL_BUILD=1 -DHAVE_NPY_CONFIG_H=1 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE=1 -D_LARGEFILE64_SOURCE=1 -DNO_ATLAS_INFO=3 -DHAVE_CBLAS -Ibuild/src.macosx-10.14-x86_64-3.8/numpy/core/src/common -Ibuild/src.macosx-10.14-x86_64-3.8/numpy/core/src/umath -Inumpy/core/include -Ibuild/src.macosx-10.14-x86_64-3.8/numpy/core/include/numpy -Ibuild/src.macosx-10.14-x86_64-3.8/numpy/distutils/include -Inumpy/core/src/common -Inumpy/core/src -Inumpy/core -Inumpy/core/src/npymath -Inumpy/core/src/multiarray -Inumpy/core/src/umath -Inumpy/core/src/npysort -Inumpy/core/src/_simd -I<venv_path>/include -I/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.8/include/python3.8 -Ibuild/src.macosx-10.14-x86_64-3.8/numpy/core/src/common -Ibuild/src.macosx-10.14-x86_64-3.8/numpy/core/src/npymath -c numpy/core/src/multiarray/dragon4.c -o build/temp.macosx-10.14-x86_64-3.8/numpy/core/src/multiarray/dragon4.o -MMD -MF build/temp.macosx-10.14-x86_64-3.8/numpy/core/src/multiarray/dragon4.o.d -msse3 -I/System/Library/Frameworks/vecLib.framework/Headers" failed with exit status 1
    ----------------------------------------
    ERROR: Failed building wheel for numpy
```
Workaround: use the [Docker instructions](#getting-started-(with-docker))

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
