# autoencoders-neat  for  anomaly  detection(ANAD)

### Utilizing Autoencoders created by neuroevolution for anomaly detection

### Description 📝

The proposed method ANAD is designing a topology of autoencoders for anomaly detection.

### What it can do? 👀

* **Construct novel autoencoder's architecture** using neuroevolution based on NEAT algorithm.
* Allow an **unsupervised machine learning algorithm** to make decisions that mark the threshold between normal and
  anomalous data instances.
* **Finds anomalies** in predictive maintenance dataset based on configuration parameters

### Requirements ✅

* **Anaconda** enviroment with Python 3.8.x (to run ANAD script).
* **Setup of project** `pip install .`

### Documentation 📘

This code's paper is currently in the writing stage. If you can't wait, I recommend checking out a similar research, in
which neural architecture search (NAS) was utilized instead of neuroevolution.

* [Our related work](https://github.com/SasoPavlic/AutoDaedalus)

### Usage 🔨

##### Changing directory

`cd examples/autoencoder`

##### Configurating parameters

Configure `evolve-autoencoder.cfg` according to your needs.

##### Running ANAD script

`python evolve-autoencoder.py`

### HELP ⚠️

**saso.pavlic@student.um.si**

## Acknowledgments 🎓

* ANAD was developed under the supervision
  of [doc. dr Sašo Karakatič](https://ii.feri.um.si/en/person/saso-karakatic-2/)
  at [University of Maribor](https://www.um.si/en/home-page/).

* This code is a fork of [palmettos](https://github.com/palmettos/autoencoders-neat)
  and [CodeReclaimers](https://github.com/CodeReclaimers/neat-python). I am grateful that the authors chose to
  open-source their work for future use.
