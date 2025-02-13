#!/bin/bash

source /apps/local/anaconda2023/bin/activate sfed

python --version

STARTTIME=$(LANG=en_us_88591; date)

COMMANDBASE="python tasks/cls-fedbase-train.py"

#-------------------------------------------------------------------------------

SEED=73
SAVEROOT=/home/joseph.benjamin/WERK/fed-cvpr/thesis_hypes/REBUTE-ATTK/cifar10-rand/
EXP_JSON=/home/joseph.benjamin/WERK/fed-cvpr/thesis_hypes/main-frame/configs/automaton/cifar10-rand.json

export CUDA_VISIBLE_DEVICES=1

##==============================================================================

ATTK_ROOT=/home/joseph.benjamin/WERK/fed-cvpr/thesis_hypes/main-frame/configs/automaton/attacks/
DFEN_ROOT=/home/joseph.benjamin/WERK/fed-cvpr/thesis_hypes/main-frame/configs/automaton/defense/

# ATTACKS=("no_attack" "alie" "fang" "ipm" "labelflip" "mimic" "scale") #"rop"
ATTACKS=("minmaxsum") #"rop"

DEFENCES=("copod" "no_defense" "krum" "cwtm" "geomrfa"
          "clipping" "cc_randbuck" "cc_seqbuck" "ties_merge"
          "fedrise_v2"
          "huberlossmin" "norm_grad_agg" "fldetector")


# Loop through each directory and append it to the root directory
for atknm in "${ATTACKS[@]}"; do

    for dfenm in "${DEFENCES[@]}"; do
        $COMMANDBASE --load-json $EXP_JSON --checkpoint-dir "$SAVEROOT/$dfenm/$atknm/" --defense-json "$DFEN_ROOT/$dfenm.json" --attack-json "$ATTK_ROOT/$atknm.json"  --seed $SEED

    done
done

echo $STARTTIME , $(LANG=en_us_88591; date)