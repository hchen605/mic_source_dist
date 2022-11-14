# !/bin/bash

set -euo pipefail

downloaded_model=assets/ESResNeXtFBSP_AudioSet.pt
model_link=https://github.com/AndreyGuzhov/ESResNeXt-fbsp/releases/download/v0.1/ESResNeXtFBSP_AudioSet.pt
train_config=protocols/dist_regression/esresnextfbsp-dist-phase3-seen-train.json
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
    if [ ! -f $downloaded_model ]; then
        echo "Stage 0: download model to $downloaded_model"
        mkdir -p `dirname $downloaded_model`
        wget -O $downloaded_model $model_link
    fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Train model using regression"
    
    pretrained=$downloaded_model
    config=$train_config
    train
fi

execution_time=$[$SECONDS-$start_time]

echo "======================================================"
echo "Total execution time: `utils/timer.pl ${execution_time}`"
echo "======================================================"
