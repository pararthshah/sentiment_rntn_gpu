# Location of the CUDA Toolkit
CUDA_PATH       ?= /Developer/NVIDIA/CUDA-6.5

OSUPPER = $(shell uname -s 2>/dev/null | tr "[:lower:]" "[:upper:]")
OSLOWER = $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

OS_SIZE    = $(shell uname -m | sed -e "s/x86_64/64/" -e "s/armv7l/32/" -e "s/aarch64/64/")
OS_ARCH    = $(shell uname -m)
ARCH_FLAGS =

DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
	XCODE_GE_5 = $(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5)
endif

# Take command line flags that override any of these settings
ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif
ifeq ($(ARMv7),1)
	OS_SIZE    = 32
	OS_ARCH    = armv7l
	ARCH_FLAGS = -target-cpu-arch ARM
endif
ifeq ($(aarch64),1)
	OS_SIZE    = 64
	OS_ARCH    = aarch64
	ARCH_FLAGS = -target-cpu-arch ARM
endif

# Common binaries
ifneq ($(DARWIN),)
ifeq ($(XCODE_GE_5),1)
  GCC ?= clang++
else
  GCC ?= g++
endif
else
ifeq ($(ARMv7),1)
  GCC ?= arm-linux-gnueabihf-g++
else
  GCC ?= g++
endif
endif
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)

# internal flags
NVCCFLAGS   := -m${OS_SIZE} ${ARCH_FLAGS}
CCFLAGS     := -stdlib=libstdc++
LDFLAGS     :=

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

# OS-specific build flags
ifneq ($(DARWIN),)
  LDFLAGS += -rpath $(CUDA_PATH)/lib
  CCFLAGS += -arch $(OS_ARCH)
else
  ifeq ($(OS_ARCH),armv7l)
    ifeq ($(abi),androideabi)
      NVCCFLAGS += -target-os-variant Android
    else
      ifeq ($(abi),gnueabi)
        CCFLAGS += -mfloat-abi=softfp
      else
        # default to gnueabihf
        override abi := gnueabihf
        LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
        CCFLAGS += -mfloat-abi=hard
      endif
    endif
  endif
endif

ifeq ($(ARMv7),1)
ifneq ($(TARGET_FS),)
GCCVERSIONLTEQ46 := $(shell expr `$(GCC) -dumpversion` \<= 4.6)
ifeq ($(GCCVERSIONLTEQ46),1)
CCFLAGS += --sysroot=$(TARGET_FS)
endif
LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)
endif
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  :=
LIBRARIES :=

################################################################################

# Gencode arguments
ifeq ($(OS_ARCH),armv7l)
SMS ?= 20 30 32 35 37 50
else
SMS ?= 11 20 30 35 37 50
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

LIBRARIES += -lcublas

################################################################################

# Target rules
all: build

build: rntn

cuda_interface.o: cuda_interface.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

sentiment_kernels.o: sentiment_kernels.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

tree.o: tree.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

training.o: training.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

rntn.o: rntn.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

rntn: cuda_interface.o sentiment_kernels.o tree.o training.o rntn.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	./rntn

clean:
	rm -f rntn *.o

clobber: clean
