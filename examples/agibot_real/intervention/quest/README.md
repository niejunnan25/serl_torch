# VLM_RDP 
This repository provides support for the VLM_RDP framework.

## 遥操作部分
For hardware and installation dependencies, see (https://github.com/rail-berkeley/oculus_reader)

## RDP
see (https://github.com/xiaoxiaoxh/reactive_diffusion_policy)

## Details
### data collection
Data collection details:[README.md](Teleoperation/README.md)

### tarin diffusion policy:
```
cd DP
./ train_dp.sh
```

### train reactive diffusion policy
```
cd DP
./train_trdp.sh
```

### some import file
```
dp config: 
	config/train_transformer_diffusion.yaml
	config/task/real_grasp_image_gelsight_emb_dp_absolute_12fps.yaml


rdp config:
	at:
	config/train_at.yaml
	config/real_grasp_image_gelsight.yaml
	config/at/skill_grasp.yaml
	rdp:
	config/train_diffusion.yaml
	config/task/real_grasp_image_gelsight_emb_ldp_24fps.yaml
```

