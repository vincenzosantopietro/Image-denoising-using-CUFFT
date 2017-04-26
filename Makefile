IDIR = $(CUDA_HOME)/samples/common/inc/
CC=nvcc
CFLAGS+=-I$(IDIR) `pkg-config --cflags opencv`
LDFLAGS+= -lcufft -lopencv_imgproc -lopencv_core -lopencv_highgui

ODIR=obj
dummy_build_folder := $(shell mkdir -p $(ODIR))

_OBJ = CufftNoiseReduction.o  
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)


convolution: $(OBJ)
	nvcc -o $@ $^ $(CFLAGS) $(LDFLAGS)
.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
 
