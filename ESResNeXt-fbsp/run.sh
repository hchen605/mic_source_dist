# !/bin/bash

set -euo pipefail

downloaded_model=assets/ESResNeXtFBSP_AudioSet.pt
pretrain_config=protocols/dist_classification/esresnextfbsp-dist-train.json
train_config=protocols/dist_regression/esresnextfbsp-dist-regression-train.json
visdom_port=8097
stage=0
stop_stage=999

. utils/parse_options.sh

start_time=$SECONDS

train () {
    echo "    config=$config"
    echo "    visdom_port=$visdom_port"
    visdom -logging_level WARN -port $visdom_port 2>&1 >/dev/null & visdom_pid=$!
    echo "    Visdom runs on PID=$visdom_pid"
    sleep 1
    [ -z `ps -p $visdom_pid -o pid=` ] && echo "Error: Failed to start visdom" && exit 1

    if [ -z $pretrained ]; then
        extra_opts=
    else
        echo "    pertrained=$pretrained"
        extra_opts="--pretrained $pretrained"
    fi
    python main.py \
        --config $config \
        --visdom-port $visdom_port $extra_opts || { kill $visdom_pid; exit 1; }
    kill $visdom_pid
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Pretraining using classification"
    rm -rf weights/MicClassification_PTINAS_ESRNXFBSP-dist
    
    pretrained=$downloaded_model
    config=$pretrain_config; train
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Train model using regression"
    rm -rf weights/MicClassification_PTINAS_ESRNXFBSP_R-dist

    pretrain_dir=weights/MicClassification_PTINAS_ESRNXFBSP-dist
    pretrained=$pretrain_dir/`ls $pretrain_dir | head -n 1`
    
    config=$train_config; train
fi

execution_time=$[$SECONDS-$start_time]

echo "======================================================"
echo "Total execution time: `utils/timer.pl ${execution_time}`"
echo "======================================================"
