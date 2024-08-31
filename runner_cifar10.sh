#!/bin/bash

source /apps/local/anaconda2023/bin/activate sfed

python --version

STARTTIME=$(LANG=en_us_88591; date)

COMMANDBASE="python tasks/cls-fedbase-train.py"

#-------------------------------------------------------------------------------

SEED=73
SAVEROOT=/home/joseph.benjamin/WERK/fed-cvpr/thesis_hypes/DEBUGGER/cifar10-rand/
EXP_JSON=/home/joseph.benjamin/WERK/fed-cvpr/thesis_hypes/main-frame/configs/automaton/cifar10-rand.json

export CUDA_VISIBLE_DEVICES=0

##==============================================================================

ATTK_ROOT=/home/joseph.benjamin/WERK/fed-cvpr/thesis_hypes/main-frame/configs/automaton/attacks/
DFEN_ROOT=/home/joseph.benjamin/WERK/fed-cvpr/thesis_hypes/main-frame/configs/automaton/defense/

ATTACKS=("alie" "fang" "ipm" "labelflip" "mimic" "scale" "no_attack") #"rop"
# ATTACKS=("scale" "no_attack") #"rop"

# Loop through each directory and append it to the root directory
for atknm in "${ATTACKS[@]}"; do

    # dfenm="no_defense"
    # dfenm="krum"
    # dfenm="copod"
    # dfenm="cwtm"
    # dfenm="geomrfa"

    # dfenm="clipping"
    # dfenm="cc_randbuck"
    # dfenm="cc_seqbuck"
    # dfenm="ties_merge"

    dfenm="fedrise_v2"

    $COMMANDBASE --load-json $EXP_JSON --checkpoint-dir "$SAVEROOT/$dfenm/$atknm/" --defense-json "$DFEN_ROOT/$dfenm.json" --attack-json "$ATTK_ROOT/$atknm.json"  --seed $SEED

done

echo $STARTTIME , $(LANG=en_us_88591; date)