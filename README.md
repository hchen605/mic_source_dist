# mic_source_dist

## Docker Environment

#### Step 1: build docker environment

```bash
docker build -t hchen605/mic_source_dist .
```

#### Step 2: Launch docker environment

```bash
docker run --shm-size=1g -v <dataset_path>:/home/speech -it --rm --privileged --gpus all -w /home/mic_source_dist hchen605/mic_source_dist:latest
```

## Training

#### ESResNext

```bash
cd ESResNext-fbsp
./run.sh [options]
```

##### Options

* `--train_config <config>`
  
  Default: `protocols/dist_regression/esresnextfbsp-dist-phase3-seen-train.json`
  
  The configuration files of the training data. The examples of configuration files are put in [ESResNext-fbsp/protocols](ESResNext-fbsp/protocols).

* `--visdom_port <visdom prot number>`
  
  Default: `8097`
  
  The port where the visdom runs.

* `--downloaded_model <downloaded_model>`, `--model_link <model_link>`
  
  Default: `assets/ESResNeXtFBSP_AudioSet.pt`, `https://github.com/AndreyGuzhov/ESResNeXt-fbsp/releases/download/v0.1/ESResNeXtFBSP_AudioSet.pt`
  
  If the `downloaded_model` doesn't exist, automatically download model from `<model_link>`.

* `--stage <start_stage>`, `--stop_stage <stop_stage>`
  
  Default: `0, 999`
  
  Runing stages of the run script
  
  * Stage 0: Download model from `model_link` to `downloaded_model`. Automatically skip this stage if `downloaded_model` exists.
  
  * Stage 1: Start the visdom process and run the training script.

## ## Testing

## Trouble Shooting

#### ESResNeXt

* Port  8097 is in use
  
  Try to change another visdom port using `--visdom_port`. Or try `pkill visdom`

* CUDA out of memory

  The default configuration requires roughly 32434 MiB of GPU. Make sure there are at least 33000 MiB of it. If you have enough GPU but still get the error, try the following command to reset the GPU.
  ```
    # Please replace the <gpu id> to the gpu index (0, 1, 2, 3, etc)
    export CUDA_VISIBLE_DEVICES=<gpu id>
  ```
  If you don't have enough GPU, try run the script with a smaller batch size. You can change the batch size by modifying [the configuration files](ESResNeXt-fbsp/protocols)

## Current result:
