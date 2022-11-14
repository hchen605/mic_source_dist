# !/bin/bash

set -euo pipefail

test_config=protocols/dist_regression/esresnextfbsp-dist-phase3-seen-test.json
savedir=weights/seen/MicClassification_PTINAS_ESRNXFBSP_R-dist
trained_model=

. utils/parse_options.sh

start_time=$SECONDS

if [ -z $trained_model ]; then
    trained_model=$savedir/`ls $savedir | head -n 1`
else
    trained_model=$savedir/$trained_model
fi

echo "Using config=$test_config"
echo "Using model=$trained_model"

python main.py \
    --pretrained $trained_model \
    --config $test_config

execution_time=$[$SECONDS-$start_time]

echo "======================================================"
echo "Total execution time: `utils/timer.pl ${execution_time}`"
echo "======================================================"
