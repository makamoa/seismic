#!/bin/bash

#python train.py --model=unet --problem=firstbreak --noise_type=-1 --noise_scale=0.0 --device=1 --epochs=15
#
python train.py --model=swin --problem=denoise --noise_type=5 --noise_scale=1.0 --device=0 --epochs=100 --pretrained=swin_denoise_noisetype_5_noisescale_1.0_attack_none_pretrained_True.pkl
python train.py --model=swin --problem=denoise --noise_type=5 --noise_scale=2.0 --device=0 --epochs=200
python train.py --model=swin --problem=denoise --noise_type=4 --noise_scale=1.0 --device=0 --epochs=200
python train.py --model=swin --problem=denoise --noise_type=4 --noise_scale=2.0 --device=0 --epochs=200
python train.py --model=swin --problem=denoise --noise_type=6 --noise_scale=1.0 --device=0 --epochs=200
python train.py --model=swin --problem=denoise --noise_type=6 --noise_scale=2.0 --device=0 --epochs=200

python train.py --model=swin --problem=firstbreak --noise_type=5 --noise_scale=1.0 --device=0 --epochs=100
#python train.py --model=swin --problem=firstbreak --noise_type=5 --noise_scale=2.0 --device=0 --epochs=50
#python train.py --model=swin --problem=firstbreak --noise_type=4 --noise_scale=1.0 --device=0 --epochs=50
#python train.py --model=swin --problem=firstbreak --noise_type=4 --noise_scale=2.0 --device=0 --epochs=50
#python train.py --model=swin --problem=firstbreak --noise_type=6 --noise_scale=1.0 --device=0 --epochs=50
#python train.py --model=swin --problem=firstbreak --noise_type=6 --noise_scale=2.0 --device=0 --epochs=50
###
#python train.py --model=unet --problem=firstbreak --noise_type=6 --noise_scale=1.0 --device=0 --epochs=30 --pretrained=unet_denoise_noisetype_6_noisescale_1.0_attack_none_pretrained_False.pkl
#python train.py --model=unet --problem=firstbreak --noise_type=6 --noise_scale=2.0 --device=0 --epochs=30 --pretrained=unet_denoise_noisetype_6_noisescale_2.0_attack_none_pretrained_False.pkl
#python train.py --model=swin --problem=firstbreak --noise_type=6 --noise_scale=1.0 --device=0 --epochs=30 --pretrained=swin_denoise_noisetype_6_noisescale_1.0_attack_none_pretrained_False.pkl
#python train.py --model=swin --problem=firstbreak --noise_type=6 --noise_scale=2.0 --device=0  --epochs=30 --pretrained=swin_denoise_noisetype_6_noisescale_2.0_attack_none_pretrained_False.pkl
##
#python train.py --model=unet --problem=firstbreak --noise_type=1 --noise_scale=0.25 --device=0 --epochs=30 --attack=fgsm
#python train.py --model=unet --problem=firstbreak --noise_type=1 --noise_scale=0.5 --device=0 --epochs=30 --attack=fgsm
#python train.py --model=unet --problem=firstbreak --noise_type=1 --noise_scale=1.0 --device=0 --epochs=30 --attack=fgsm
#python train.py --model=unet --problem=firstbreak --noise_type=1 --noise_scale=2.0 --device=0  --epochs=30 --attack=fgsm
##
#python train.py --model=unet --problem=firstbreak --noise_type=2 --noise_scale=0.25 --device=0 --epochs=30 --attack=fgsm
#python train.py --model=unet --problem=firstbreak --noise_type=2 --noise_scale=0.5 --device=0 --epochs=30 --attack=fgsm
#python train.py --model=unet --problem=firstbreak --noise_type=2 --noise_scale=1.0 --device=0 --epochs=30 --attack=fgsm
#python train.py --model=unet --problem=firstbreak --noise_type=2 --noise_scale=2.0 --device=0  --epochs=30 --attack=fgsm
##
#python train.py --model=unet --problem=firstbreak --noise_type=3 --noise_scale=0.25 --device=0 --epochs=30 --attack=fgsm
#python train.py --model=unet --problem=firstbreak --noise_type=3 --noise_scale=0.5 --device=0 --epochs=30 --attack=fgsm
#python train.py --model=unet --problem=firstbreak --noise_type=3 --noise_scale=1.0 --device=0 --epochs=30 --attack=fgsm
#python train.py --model=unet --problem=firstbreak --noise_type=4 --noise_scale=2.0 --device=0  --epochs=30 --attack=fgsm --epsilon=0.0125 --prefix=attack_0.0125
#python train.py --model=unet --problem=firstbreak --noise_type=4 --noise_scale=2.0 --device=0  --epochs=30 --attack=fgsm --epsilon=0.025 --prefix=attack_0.0250
#python train.py --model=unet --problem=firstbreak --noise_type=4 --noise_scale=2.0 --device=0  --epochs=30 --attack=fgsm --epsilon=0.05 --prefix=attack_0.0500
#python train.py --model=unet --problem=firstbreak --noise_type=4 --noise_scale=2.0 --device=0  --epochs=30 --attack=fgsm --epsilon=0.1 --prefix=attack_0.1000