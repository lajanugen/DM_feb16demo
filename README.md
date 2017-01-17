
This repository implements a dialog manager for the DSTC2 restaurant domain data.

The DM takes a sequence of past user and system utterances and predicts the next system utterance.

Data: Download DSTC2 data from ```http://camdial.org/~mh521/dstc/```. Create a folder named ```data``` in the same level as the src folder. Organize the downloaded data so that the data folder will have sub-directories named config, train and test where the train and test directories will consist of the actual data ("Mar*") from the relevant zip files.

Training

The train/lstm/train_lstm.lua script is used to for training. Adjust the hyperparameter values at the beginning of the script as necessary.

Run the script as follows

```
th train_lstm.lua
```

This prints the training progress along with train and validation performance to monitor the progress. At the end of the run (when the maximum number of epochs are reached), it outputs the best validation accuracy and the corresponding epoch. Model checkpoints are saved in `cache/dump`.

Parameters of the best model will be logged into a file named `res/final\_\*.txt`. This file will be specified as input to the evaluation script.

Evaluation (demo mode)

Please see `DM/prod/README.md` for instructios on evaluation.
