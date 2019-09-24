# Use the Anaconda Docker image
FROM continuumio/anaconda3:5.3.0

# Create Python 3.5 environment and activate it
# This will use the enviroment.yml file specification
ADD environment.yml /tmp/environment.yml
RUN conda env create -f=/tmp/environment.yml
RUN echo "source activate silcam" > ~/.bashrc
ENV PATH /opt/conda/envs/silcam/bin:$PATH

# Install build tools, gcc etc
RUN apt-get update
RUN apt-get install -y build-essential

# Add the Pysilcam source directory to the container in order to install Pysilcam
ADD . /silcam

# Assume the source code is mapped to /silcam
WORKDIR /silcam

# Install PySilcam dependencies
RUN python setup.py develop

# Run the PySilCam tests as default entrypoint
ENTRYPOINT ["python", "setup.py", "test"]
