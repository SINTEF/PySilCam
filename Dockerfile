# Use the Anaconda Docker image
#FROM continuumio/anaconda3:5.3.0
FROM continuumio/miniconda3

# Install build tools, gcc etc
RUN apt-get update && apt-get install -y build-essential vim htop procps x11vnc xvfb libsdl-ttf2.0-0

# Create Python 3.5 environment and activate it
# This will use the enviroment.yml file specification
RUN echo "source activate silcam" > ~/.bashrc
ENV PATH /opt/conda/envs/silcam/bin:$PATH
ADD environment.yml /tmp/environment.yml
RUN conda env create -f=/tmp/environment.yml

# VNC server for GUI, to be run on port 5920
ENV DISPLAY :20
EXPOSE 5920

# Make Python not create .pyc files
ENV PYTHONDONTWRITEBYTECODE FALSE

# Assume that test data is mounted into the testdata path
ENV UNITTEST_DATA_PATH /testdata/unittest-data
ENV SILCAM_MODEL_PATH /testdata/tflmodel/particle-classifier.tfl

# Supress the usage of a display for unit tests
ENV MPLBACKEND Agg

# Add the Pysilcam source directory to the container in order to install Pysilcam
ADD . /silcam

# Assume the source code is mapped to /silcam
WORKDIR /silcam

# Install PySilcam dependencies
RUN python setup.py develop

# Run the PySilCam tests as default entrypoint
CMD ["python", "setup.py", "test_noskip"]
