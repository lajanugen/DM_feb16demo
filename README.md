
This repository implements a dialog manager for the DSTC2 restaurant domain data.

The DM takes a sequence of past user and system utterances and predicts the next system utterance.

Training

The `train/lstm/train_lstm.lua1 script is used to for training. Adjust the hyperparameter values at the beginning of the script as necessary.

Run the script as follows

```
th train_lstm.lua
```

This prints the training progress along with train and validation performance to monitor the progress. At the end of the run (when the maximum number of epochs are reached), it outputs the best validation accuracy and the corresponding epoch. Model checkpoints are saved in `cache/dump`.

Parameters of the best model will be logged into a file named `res/final_*.txt`. This file will be specified as input to the evaluation script.

Evaluation (demo mode)

Please see `DM/prod/README.md` for instructios on evaluation.
