# Demo of DM

This module will take a JSON input string from stdin and produce the ouput as a JSON string to stdout. The input comes from the previous block which is the NLU value predictor (which is invoked after NLU das predictor) and the ouput will go to NLG module.

To run the module, specify the settings file from the training procedure for parameter `opt.config` in `DM/prod/src/dm_svc.lua`.

Then to run the demo, do
```
cd dm/prod/lstm/
rm db.json; th serve_dm.lua < sample_input.json
```

where `sample_input.json` is an example utterance with annotations.

The history of the current conversation is logged into `db.json`.

Note: for a new dialogue converstaion, we would need to delete `db.json` in order to get DM to work correctly.
Within a single conversation, DM will use this file to save information it needs across turns.

