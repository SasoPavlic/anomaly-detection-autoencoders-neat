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

**10thJubilee International Conference on Computational Cybernetics and Cyber-Medical Systems:**
[Predictive maintenance of pumping system bearings with evolved autoencoders](https://www.sasopavlic.com/publication/predictive-maintenance-of-pumping-system-bearings-with-evolved-autoencoders/)

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
  
  # Cite us
Are you using ANAD in your project or research? Please cite us!
### Plain format
```
S. Pavlič, N. Young, and S. Karakatič, “Predictive maintenance of pumping system bearings with evolved autoencoders,” in 2022 IEEE 10th Jubilee International Conference on Computational Cybernetics and Cyber-Medical Systems (ICCC), Jul. 2022, pp. 000149–000154. doi: 10.1109/ICCC202255925.2022.9922678.
```
### Bibtex format
```
@inproceedings{pavlic_predictive_2022,
	title = {Predictive maintenance of pumping system bearings with evolved autoencoders},
	doi = {10.1109/ICCC202255925.2022.9922678},
	eventtitle = {2022 {IEEE} 10th Jubilee International Conference on Computational Cybernetics and Cyber-Medical Systems ({ICCC})},
	pages = {000149--000154},
	booktitle = {2022 {IEEE} 10th Jubilee International Conference on Computational Cybernetics and Cyber-Medical Systems ({ICCC})},
	author = {Pavlič, Sašo and Young, Nicholas and Karakatič, Sašo},
	date = {2022-07},
	keywords = {anomaly detection, neuroevolution, autoen-coder, Predictive maintenance},
	file = {IEEE Xplore Abstract Record:C\:\\Users\\sasop\\Zotero\\storage\\A7LRPXY8\\9922678.html:text/html;IEEE Xplore Full Text PDF:C\:\\Users\\sasop\\Zotero\\storage\\Q2W2H8MQ\\Pavlič et al. - 2022 - Predictive maintenance of pumping system bearings .pdf:application/pdf},
}
```
### RIS format
```
TY  - CONF
TI  - Predictive maintenance of pumping system bearings with evolved autoencoders
AU  - Pavlič, Sašo
AU  - Young, Nicholas
AU  - Karakatič, Sašo
T2  - 2022 IEEE 10th Jubilee International Conference on Computational Cybernetics and Cyber-Medical Systems (ICCC)
DA  - 2022/07//
PY  - 2022
DO  - 10.1109/ICCC202255925.2022.9922678
DP  - IEEE Xplore
SP  - 000149
EP  - 000154
L1  - https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=9922678&ref=aHR0cHM6Ly9pZWVleHBsb3JlLmllZWUub3JnL2RvY3VtZW50Lzk5MjI2Nzg=
L2  - https://ieeexplore.ieee.org/document/9922678
KW  - anomaly detection
KW  - neuroevolution
KW  - autoen-coder
KW  - Predictive maintenance
```
