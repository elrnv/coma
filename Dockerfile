FROM ubuntu:latest

ENV CUDA=cpu

RUN apt-get update
RUN apt-get install -y build-essential python3-pip libboost-dev python3.6-venv
RUN apt-get install -y cmake
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html && \
    python3 -m pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html && \
    python3 -m pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html && \
    python3 -m pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html && \
    python3 -m pip install torch-geometric
RUN python3 -m pip install openmesh && \
    python3 -m pip install tqdm

RUN apt-get install -y git

RUN git clone https://github.com/MPI-IS/mesh.git /root/pytorch/mesh
RUN cd /root/pytorch/mesh && \
    python3 -m venv --system-site-packages --copies mesh_venv && \
    . mesh_venv/bin/activate && \
    BOOST_INCLUDE_DIRS=/usr/lib/include make all

RUN apt-get install -y vim
RUN apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev


ADD . /data/

