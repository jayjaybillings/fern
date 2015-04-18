CXX = g++

CXXFLAGS = -g -Wall -O3

SRCS = \
	src/main.cpp \
	src/FERNIntegrator.cpp \
	src/Network.cpp \
	src/IntegrationData.cpp \
	src/kernels.cpp

OBJS = $(SRCS:.cpp=.o)

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
