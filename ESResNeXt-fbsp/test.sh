# !/bin/bash

set -euo pipefail

ESResNeXt_config=protocols/dist_regression/esresnextfbsp-dist-regression-test_multi.json
savedir=weights/MicClassification_PTINAS_ESRNXFBSP_R-dist
pretrained=

. utils/parse_options.sh

start_time=$SECONDS

if [ -z $pretrained ]; then
    pretrained=$savedir/`ls $savedir | head -n 1`
else
    pretrained=$savedir/$pretrained
fi

echo "Using config=$ESResNeXt_config"
echo "Using model=$pretrained"

python main.py \
    --pretrained $pretrained \
    --config $ESResNeXt_config

execution_time=$[$SECONDS-$start_time]

echo "======================================================"
echo "Total execution time: `utils/timer.pl ${execution_time}`"
echo "======================================================"
