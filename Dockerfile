FROM python:3

WORKDIR /app

# Install Graphviz
RUN apt-get update && apt-get install -y graphviz

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install graphviz

# Create directories and copy files
RUN mkdir datasets
RUN mkdir examples
RUN mkdir examples/cardiovascular-risk
RUN mkdir neat

COPY datasets/CVD_formated.csv /app/datasets/CVD_formated.csv
COPY examples/cardiovascular-risk/anomalyDetection.py /app/examples/cardiovascular-risk/anomalyDetection.py
COPY examples/cardiovascular-risk/data-loader.py /app/examples/cardiovascular-risk/data-loader.py
COPY examples/cardiovascular-risk/evolve-autoencoder.cfg /app/examples/cardiovascular-risk/evolve-autoencoder.cfg
COPY examples/cardiovascular-risk/evolve-autoencoder.py /app/examples/cardiovascular-risk/evolve-autoencoder.py
COPY examples/cardiovascular-risk/visualize.py /app/examples/cardiovascular-risk/visualize.py
COPY neat /app/neat

COPY setup.py /app/setup.py
RUN pip3 install .

# Other commands
CMD tree
RUN python -c "import neat ; print(neat.population)" >> neat-location.info
