# Image denoising using CUFFT

A parallel implementation for image denoising on a Nvidia GPU using [Cuda][cu] and the [cuFFT Library][df1] [1]
The sofware:
  - Automatically selects the most powerful GPU (in case of a multi-GPU system)
  - Executes denoising
  - Saves output as text file and image

# Authors

  - Vincenzo Santopietro <<vincenzo.santopietro@uniparthenope.it>>
  - Francesco Battistone <<francesco.battistone@uniparthenope.it>>

### Tech

Dillinger uses a number of open source projects to work properly:

* [AngularJS] - HTML enhanced for web apps!
* [Ace Editor] - awesome web-based text editor
* [markdown-it] - Markdown parser done right. Fast and easy to extend.
* [Twitter Bootstrap] - great UI boilerplate for modern web apps
* [node.js] - evented I/O for the backend
* [Express] - fast node.js network app framework [@tjholowaychuk]
* [Gulp] - the streaming build system
* [Breakdance](http://breakdance.io) - HTML to Markdown converter
* [jQuery] - duh

And of course Dillinger itself is open source with a [public repository][dill]
 on GitHub.

### Installation
The software requires [OpenCV](http://opencv.org/opencv-v2-4-2-released.html) v2.4.x to run.

Install the dependencies and compile the Makefile.

```sh
$ cd image-denoising-using-cufft
$ make
```

### Plugins

| Plugin | README |
| ------ | ------ |
| OpenCV | [https://github.com/opencv/opencv/blob/2.4/README.md] [PlDb] |
| cuFFT | [http://docs.nvidia.com/cuda/cufft/#axzz4f6kyGEZu] [PlGd] |


### Development

Want to contribute? Great!



### Todos

 - Add an input parser 
 - Clean code

License
----

GNUv3.0


**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [cu]: <https://developer.nvidia.com/cuda-zone>
   [df1]: <https://developer.nvidia.com/cufft>
   
   >
