FROM nvcr.io/nvidia/pytorch:22.09-py3
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y htop
RUN apt-get install -y nano
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN pip3 install pandas==2.0.3 --user
RUN pip3 install numpy==1.24.4 --user
RUN pip3 install albumentations==1.3.1 --user
RUN pip3 install tqdm==4.66.1 --user
RUN pip3 install scikit-image==0.21.0 --user
RUN pip3 install scikit-learn==1.3.2 --user
RUN pip3 install timm==0.9.8 --user
RUN pip3 install matplotlib==3.7.3 --user
ENV PATH="${PATH}:/root/.local/bin"

