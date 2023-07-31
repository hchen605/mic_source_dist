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

## Data preparation

#### Step 1: Place the wav files

Place all the wav files for training and testing into a directory. The path of the directory doesn't matter, just pass it to the code using the `root` parameter.

#### Step 2: Creat csv files contianing the waveform information

Create `train_csv`, `val_csv` and `test_csv` using the following format.
The separator for the csv files are `\t`.

```text
filename distance
<subpath of wav1 from root> <labeled distance1>
<subpath of wav2 from root> <labeled distance2>
<subpath of wav3 from root> <labeled distance3>
.
.
.
```

Take a look at the example [train_csv](data/phase3_all_seen_train.csv).

## AttCNN

#### Training

```bash
cd attcnn/fcnn
python dist_train.py [options]
```

###### Options

* `--savedir <savedir>`
  Default: `weights/AttCNN`
  
  The path where the trained model is saved

* `--train_csv <train_csv>`, `--dev_csv <dev_csv>`
  Default: `../../data/phase3_all_seen_train.csv`, `../../data/phase3_all_seen_val.csv`
  
  The path of the file containing the wavform information

* `--root <root>`
  Default: `/home/speech`
  
  The prefix of the path listed in `<train_csv>` and `<dev_csv>`

* **D4 new** `--rir_root <rir_root>`
  The path for the rir directory. Only effective when the `--timit_root` is specified.
  The rir directory contains impulse responses of the recording. We applied the rir to all the data from `timit_root`.

* **D4 new** `--timit_root <timit_root>`
  The path for the TIMIT dataset. Only effective when the `--rir_root` is specified.

#### Testing

```bash
cd attcnn/fcnn
python dist_test.py [options]
```

###### Options

* `--savedir <savedir>`
  Default: `weights/AttCNN`
  
  The path where the trained model is saved

* `--test_csv <test_csv>`
  Default: `../../data/phase3_all_seen_test.csv`
  
  The path of the file containing the wavform information

* `--root <root>`
  Default: `/home/speech`
  
  The prefix of the path listed in `<test_csv>`

## ESResNext

#### Training

**D4 New**: Please refer to [the RIR augmentation configuration](/ESResNeXt-fbsp/protocols/dist_regression/D4/jointly-train-timit.json) for jointly train with RIR augmentation. Please also refer to [the unseen room test configuration](/ESResNeXt-fbsp/protocols/dist_regression/D3/esresnextfbsp-dist-phase3-unseenroom-test.json) for the perofrmance evaluation.

```bash
cd ESResNext-fbsp
./run.sh [options]
```

###### Options

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

#### Testing

```bash
cd ESResNext-fbsp
./test.sh [options]
```

###### Options

* `--test_config <test_config>`
  Default: `protocols/dist_regression/esresnextfbsp-dist-phase3-seen-test.json`
  
  The configuration files of the testing data. The examples of configuration files are put in [ESResNext-fbsp/protocols](ESResNext-fbsp/protocols).

* `--savedir <savedir>`
  Default: `weights/seen/MicClassification_PTINAS_ESRNXFBSP_R-dist`
  
  The path where the trained model is saved

* `--trained_model <trained_model>`
  Default: The first model in `ls <savedir>`
  
  Trained model name.

#### Trouble Shooting

* Port  8097 is in use
  
  Try to change another visdom port using `--visdom_port`. Or try `pkill visdom`

* CUDA out of memory
  
  The default configuration requires roughly 32434 MiB of GPU. Make sure there are at least 33000 MiB of it. If you have enough GPU but still get the error, try the following command to reset the GPU.
  
  ```
    # Please replace the <gpu id> to the gpu index (0, 1, 2, 3, etc)
    export CUDA_VISIBLE_DEVICES=<gpu id>
  ```
  
  If you don't have enough GPU, try run the script with a smaller batch size. You can change the batch size by modifying [the configuration files](ESResNeXt-fbsp/protocols)

## Current result (MAE in meters):

|Configurations|AttCNN|ESResNeXt|
|-|-|-|
|Matched|0.0877|0.0593|
|Unseen Room|0.9982|0.8705|
|Unseen Microphone|0.5530|0.2626|
