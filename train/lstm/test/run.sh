#!/usr/bin/env bash
#
#
#

# cd cr/
bash test/sync.sh
wait
# th bin/test_dstc1_loader.lua
th train_lstm.lua
# cd -
