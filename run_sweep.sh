#!/usr/bin/env bash
set -e

CFG="configs/transfusion_minegrape_L_grapeonly.py"
GPU=1
MAXE=100
WORKROOT="work_dirs/sweeps_$(date +%m%d_%H%M)"

Ks=(64 128 256)     # num_proposals
OVs=(0.1 0.3)       # gaussian_overlap
MRs=(2 3)           # min_radius
VWS=(0.1 0.2)       # vis/cause 的 loss_weight

mkdir -p "${WORKROOT}"

i=0
for K in "${Ks[@]}"; do
  for OV in "${OVs[@]}"; do
    for MR in "${MRs[@]}"; do
      for VW in "${VWS[@]}"; do
        i=$((i+1))
        EXP="K${K}_ov${OV}_mr${MR}_vw${VW}"
        WDIR="${WORKROOT}/${EXP}"
        echo ">>> [${i}] ${EXP}"

        CUDA_VISIBLE_DEVICES=0 tools/dist_train.sh "${CFG}" ${GPU} \
          --work-dir "${WDIR}" --seed 0 --deterministic \
          --cfg-options \
          'workflow=[("train",1),("val",1)]' \
          evaluation.interval=1 \
          runner.max_epochs=${MAXE} \
          checkpoint_config.interval=${MAXE} \
          model.pts_bbox_head.num_proposals=${K} \
          model.train_cfg.pts.gaussian_overlap=${OV} \
          model.train_cfg.pts.min_radius=${MR} \
          model.pts_bbox_head.loss_vis.loss_weight=${VW} \
          model.pts_bbox_head.loss_cause.loss_weight=${VW}

      done
    done
  done
done

echo "All sweeps done."
