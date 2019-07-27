FROM nvidia/cuda:9.0-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update && apt-get install -y \
    build-essential git libatlas-base-dev libopencv-dev python-opencv \
    libcurl4-openssl-dev libgtest-dev cmake wget unzip

RUN cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib

WORKDIR /project

RUN wget https://dl.google.com/go/go1.12.7.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.12.7.linux-amd64.tar.gz
ENV PATH=$PATH:/usr/local/go/bin:/root/go/bin
ENV GOPATH=/root/go
RUN mkdir -p $GOPATH:src/gorgonia.org
WORKDIR $GOPATH/src/gorgonia.org
RUN go get -u github.com/golang/dep/cmd/dep

RUN go get -u gorgonia.org/gorgonia
WORKDIR $GOPATH/src/gorgonia.org/gorgonia
RUN go get -u gorgonia.org/cu
RUN go build -tags='cuda' .
RUN go get -u gopkg.in/cheggaaa/pb.v1

RUN go install gorgonia.org/gorgonia/cmd/cudagen

RUN mkdir -p examples/testdata/mnist
RUN cd examples/testdata/mnist && wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
RUN cd examples/testdata/mnist && wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
RUN cd examples/testdata/mnist && wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
RUN cd examples/testdata/mnist && wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

RUN cd examples/testdata/mnist && gunzip train-images-idx3-ubyte.gz
RUN cd examples/testdata/mnist && gunzip train-labels-idx1-ubyte.gz
RUN cd examples/testdata/mnist && gunzip t10k-images-idx3-ubyte.gz
RUN cd examples/testdata/mnist && gunzip t10k-labels-idx1-ubyte.gz

RUN cd examples/testdata/mnist && mv train-images-idx3-ubyte train-images.idx3-ubyte
RUN cd examples/testdata/mnist && mv train-labels-idx1-ubyte train-labels.idx1-ubyte
RUN cd examples/testdata/mnist && mv t10k-images-idx3-ubyte t10k-images.idx3-ubyte
RUN cd examples/testdata/mnist && mv t10k-labels-idx1-ubyte t10k-labels.idx1-ubyte

CMD cd examples/convnet_cuda && cudagen && go run -tags='cuda' main.go