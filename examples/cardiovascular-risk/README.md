# Optimizing Cardiovascular Disease Risk Detection with Faster Convergence via Curriculum learning Enhanced Neuroevolution 

### Utilizing Autoencoders created by neuroevolution + Curriculum learning for anomaly detection

### Description 📝

The proposed method ANAD-CL is designing a topology of autoencoders guided by fitness function.

### What it can do? 👀

* **Construct novel autoencoder's architecture** using neuroevolution based on **NEAT algorithm** + **Curriculum learning**.
* Allow an **unsupervised machine learning algorithm** to make decisions that mark the threshold between normal and
  anomalous data instances.
* **Finds anomalies** in [CDRPD dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset)

### Requirements ✅

* **Anaconda** enviroment with Python 3.8.x (to run ANAD script).
* **Setup of project** `pip install .`

### Documentation 📘

**!!!TO BE ADDED IN UPCOMING WEEKS!!!**

### Usage 🔨

##### Changing directory

`cd examples/cardiovascular-risk`

##### Configurating parameters

Configure `evolve-autoencoder.cfg` according to your needs.

##### Running ANAD script

`python evolve-autoencoder.py`

##### Running ANAD script with Docker:

```docker build --tag spartan300/anad . ```

```
docker run \
--rm \
-it \
-v $(pwd)/logs:/app/examples/cardiovascular-risk/logs \
-v $(pwd)/config:/app/examples/cardiovascular-risk/config \
-w="/app/examples/cardiovascular-risk/" \
--shm-size 8G spartan300/anad \
python evolve-autoencoder.py
```


### HELP ⚠️

**saso.pavlic@student.um.si**
