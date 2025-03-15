# seminar_fairness - Evaluation of Fairness Frameworks for Robustness against Adversarial Attacks 
This repository implements and evaluates multiple fairness frameworks applied to NLP text classification under adversarial settings. The implementations include:

- **FRAPPÉ on NLP Text Classification:** A framework that applies post-processing fairness mitigation.
- **BadFair on NLP Text Classification:** An approach to attack fairness mechanisms via data poisoning.
- **BadFair on FRAPPÉ:** Integration and cross-evaluation of the BadFair attack within the FRAPPÉ framework.

## Usage guide

### Prerequisites

Python pacakge requirements are in `requirements.txt` and need to be installed

### Workflow
There are 4 folders, 2 for each dataset, 1 with the Badfair attack and 1 without. 
Since training data poisoning and test data manipulation is needed to implement Badfair, this folder seperation was done to prevent accidental cross-contamination.
The logits are saved in the contents folder which needs to be emptied every time parameters are changed and the model is retrained.

In each folder `train_model.py` should be run first and then `evaluate.py`.
 - `train_model.py` downloads (and poisons the data) and trains the model and applies the FRAPPE framework.
 - `evaluate.py` calculates Fairness metrics and calls `postproc_fairness/fairmain.py´
 - `postproc_fairness/fairmain.py´ applies FRAPPE and calculates Fairness metrics
