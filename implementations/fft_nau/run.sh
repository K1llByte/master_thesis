#!/bin/bash

TARGET=$1
if [ -z $1 ]; then
    TARGET=fft256.xml
fi
NAU_BIN=/home/kb/Documents/MIEI/4_Grade/1_Sem/VI1/Projects/nau_bin/composerImGui.exe
XML_PATH=$(realpath $TARGET)

# Check if prime-run command is available
NVIDIA_PRIME_AVAILABLE=$(command -v prime-run)
if [ -z "$NVIDIA_PRIME_AVAILABLE" ]; then
    wine $NAU_BIN $XML_PATH
else 
    prime-run wine $NAU_BIN $XML_PATH
fi