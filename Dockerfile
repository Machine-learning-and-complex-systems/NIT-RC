# hash:sha256:dee6945231f820cc262806778097a5d72e04d8be3337c71e7570ea8c0a61960b
# The Python version can be either 3.9 or 3.10.15 (there will be slight differences in the results).
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        keras==2.10.0 \
        matplotlib==3.7.0 \
        networkx==2.8.4 \
        numpy==1.23.5 \
        tensorflow==2.10.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    filterpy==1.4.5 \
    pysindy==1.7.5 \
    scipy==1.11.3
