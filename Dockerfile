# syntax = docker/dockerfile:1.2
FROM nvidia/cuda:11.2.0-devel-ubuntu20.04 AS builder
ENV TZ=Asia/Seoul

ARG DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/apt.conf.d/docker-clean

RUN --mount=type=cache,target=/builder/var/cache/apt apt-get update
RUN --mount=type=cache,target=/builder/var/cache/apt apt-get install -yqq --no-install-recommends software-properties-common
RUN --mount=type=cache,target=/builder/var/cache/apt add-apt-repository ppa:savoury1/ffmpeg4
RUN --mount=type=cache,target=/builder/var/cache/apt apt-get update

RUN --mount=type=cache,target=/builder/var/cache/apt apt-get install -yqq --no-install-recommends build-essential curl git make cmake pkg-config
RUN --mount=type=cache,target=/builder/var/cache/apt apt-get install -yqq --no-install-recommends ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN git clone --recursive https://github.com/dmlc/decord
RUN mkdir /root/decord/build

WORKDIR /root/decord/build

COPY tmp/libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvcuvid.so

RUN cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
RUN make -j $(cat /proc/cpuinfo  | grep processor | wc -l)

FROM nvidia/cuda:11.2.0-devel-ubuntu20.04 AS runner

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt apt-get update
RUN --mount=type=cache,target=/var/cache/apt apt-get install -yqq --no-install-recommends software-properties-common
RUN --mount=type=cache,target=/var/cache/apt add-apt-repository ppa:deadsnakes/ppa
RUN --mount=type=cache,target=/var/cache/apt add-apt-repository ppa:savoury1/ffmpeg4

RUN --mount=type=cache,target=/var/cache/apt apt-get update
RUN --mount=type=cache,target=/var/cache/apt apt-get install -yqq --no-install-recommends ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
RUN --mount=type=cache,target=/var/cache/apt apt-get install -yqq --no-install-recommends libgl1-mesa-glx libglib2.0-0
RUN --mount=type=cache,target=/var/cache/apt apt-get install -yqq --no-install-recommends python3.9 python3-pip

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install setuptools

COPY --from=builder /root/decord /root/decord

WORKDIR /root/decord/python
RUN python3 setup.py install

RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchtext -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

WORKDIR /root
COPY requirements.txt /root
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r /root/requirements.txt

ARG USER_NAME=user
ARG USER_ID=1000
ARG GROUP_NAME=group
ARG GROUP_ID=1000

RUN echo "${USER_NAME}:x:${USER_ID}:${GROUP_ID}::/app:/bin/bash" >> /etc/passwd
RUN echo "domain:x:${GROUP_ID}:" >> /etc/group
RUN echo "${USER_NAME}:!:19123:0:99999:7:::" >> /etc/shadow

RUN mkdir /app
COPY . /app
RUN chown -R ${USER_NAME}:domain /app
WORKDIR /app

ENTRYPOINT [ "python3", "-m", "pytest" ]
