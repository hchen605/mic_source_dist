#!/bin/bash

set -euo pipefail

dir=assets
url=https://github.com/AndreyGuzhov/ESResNeXt-fbsp/releases/download/v0.1/ESResNeXtFBSP_AudioSet.pt

mkdir -p $dir

wget -P $dir $url
