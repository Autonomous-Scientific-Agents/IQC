# Use an official miniconda3 image as the base
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file and package files
COPY env.yml .
COPY . .

# Create conda environment and activate it
RUN conda env create -f env.yml && \
    echo "conda activate iqc-env" >> ~/.bashrc

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "iqc-env", "/bin/bash", "-c"]

# Install IQC
RUN pip install .

# Create jupyter notebook directory and example notebook if it doesn't exist
RUN mkdir -p /app/notebooks

# Expose Jupyter Lab port
EXPOSE 8888

# Set the default command to run Jupyter Lab
ENTRYPOINT ["conda", "run", "-n", "iqc-env", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"] 