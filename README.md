# Image denoising using CUFFT

A parallel implementation for image denoising on a Nvidia GPU using [Cuda][cu] and the [cuFFT Library][df1]
The sofware:
  - Automatically selects the most powerful GPU (in case of a multi-GPU system)
  - Executes denoising
  - Saves output as text file and image

# Authors

  - Vincenzo Santopietro <<vincenzosantopietro@linux.com>>
  - Francesco Battistone <<battistone.francesco@gmail.com>>

### Installation
The software requires [OpenCV](http://opencv.org/opencv-v2-4-2-released.html) v2.4.x to run.

Install the dependencies and compile the Makefile.

```sh
$ cd image-denoising-using-cufft
$ make
```
### Demo 
The software needs some parameters: 
  - Number of threads (per block, for each direction)
  - Path to the output *'.txt'* file. If the file does not exist, it's created.
  - Path to the input image
  - Path to the *'.txt'* containing the kernel. [NB kernel's size is fixed to 5x5]
 
You can run the software by typing
```sh
$ ./convolution <num_threads> <output_file_path> <image_path> <kernel_path>
```

### Plugins

| Plugin | README |
| ------ | ------ |
| OpenCV | [https://github.com/opencv/opencv/blob/2.4/README.md] [cv] |
| cuFFT | [http://docs.nvidia.com/cuda/cufft/#axzz4f6kyGEZu] [df1] |

### Version

This is still a beta version, developed for academic purposes.

### Development

Want to contribute? Great!


### Todos

 - Add an input parser 

License
----

GNUv3.0

Credits
-----
Image: https://en.wikipedia.org/wiki/Gaussian_noise 

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [cu]: <https://developer.nvidia.com/cuda-zone>
   [df1]: <https://developer.nvidia.com/cufft>
   [cv]: <http://opencv.org/opencv-v2-4-2-released.html>
   >
