CXX = gcc

CXXFLAGS = -g -Wall -O3

SRCS = \
	src/main.cu \
	src/FERNIntegrator.cu \
	src/Network.cu \
	src/Globals.cu \
	src/IntegrationData.cu \
	src/kernels.cu

OBJS = $(SRCS:.cu=.o)

all: build/fern

build/fern: $(OBJS)
	mkdir -p build
	$(LINK.cc) -o $@ $^

%.o: %.cu
	$(COMPILE.cc) -c -o $@ $^

# Convenience targets

clean:
	@rm -fv build/fern $(OBJS)

run: build/fern
	cd data; ../build/fern
