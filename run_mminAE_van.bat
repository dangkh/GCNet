@echo off 
cd baseline-mmin
set seed[0]=1234 
set seed[1]=9138
set seed[2]=86503
set seed[3]=37949
set seed[4]=22627
set seed[5]=75258
set seed[6]=94877
set seed[7]=9829
set seed[8]=47702
set seed[9]=15908

python -u train_baseline.py --mask_rate=0.6 --dataset_mode=cmumosi_multimodal  --model=mmin_AE --print_freq=10 --gpu_ids=0 --input_dim_a=250 --embd_size_a=128 --input_dim_v=1000 --embd_size_v=128 --input_dim_l=38400 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=10 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin  --batch_size=1 --output_dim=1 --seed %seed[0]% 
python -u train_baseline.py --mask_rate=0.6 --dataset_mode=cmumosi_multimodal  --model=mmin_AE --print_freq=10 --gpu_ids=0 --input_dim_a=250 --embd_size_a=128 --input_dim_v=1000 --embd_size_v=128 --input_dim_l=38400 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=10 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin  --batch_size=1 --output_dim=1 --seed %seed[1]% 
python -u train_baseline.py --mask_rate=0.6 --dataset_mode=cmumosi_multimodal  --model=mmin_AE --print_freq=10 --gpu_ids=0 --input_dim_a=250 --embd_size_a=128 --input_dim_v=1000 --embd_size_v=128 --input_dim_l=38400 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=10 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin  --batch_size=1 --output_dim=1 --seed %seed[2]% 
python -u train_baseline.py --mask_rate=0.6 --dataset_mode=cmumosi_multimodal  --model=mmin_AE --print_freq=10 --gpu_ids=0 --input_dim_a=250 --embd_size_a=128 --input_dim_v=1000 --embd_size_v=128 --input_dim_l=38400 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=10 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin  --batch_size=1 --output_dim=1 --seed %seed[3]% 
python -u train_baseline.py --mask_rate=0.6 --dataset_mode=cmumosi_multimodal  --model=mmin_AE --print_freq=10 --gpu_ids=0 --input_dim_a=250 --embd_size_a=128 --input_dim_v=1000 --embd_size_v=128 --input_dim_l=38400 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=10 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin  --batch_size=1 --output_dim=1 --seed %seed[4]% 
python -u train_baseline.py --mask_rate=0.6 --dataset_mode=cmumosi_multimodal  --model=mmin_AE --print_freq=10 --gpu_ids=0 --input_dim_a=250 --embd_size_a=128 --input_dim_v=1000 --embd_size_v=128 --input_dim_l=38400 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=10 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin  --batch_size=1 --output_dim=1 --seed %seed[5]% 
python -u train_baseline.py --mask_rate=0.6 --dataset_mode=cmumosi_multimodal  --model=mmin_AE --print_freq=10 --gpu_ids=0 --input_dim_a=250 --embd_size_a=128 --input_dim_v=1000 --embd_size_v=128 --input_dim_l=38400 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=10 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin  --batch_size=1 --output_dim=1 --seed %seed[6]% 
python -u train_baseline.py --mask_rate=0.6 --dataset_mode=cmumosi_multimodal  --model=mmin_AE --print_freq=10 --gpu_ids=0 --input_dim_a=250 --embd_size_a=128 --input_dim_v=1000 --embd_size_v=128 --input_dim_l=38400 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=10 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin  --batch_size=1 --output_dim=1 --seed %seed[7]% 
python -u train_baseline.py --mask_rate=0.6 --dataset_mode=cmumosi_multimodal  --model=mmin_AE --print_freq=10 --gpu_ids=0 --input_dim_a=250 --embd_size_a=128 --input_dim_v=1000 --embd_size_v=128 --input_dim_l=38400 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=10 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin  --batch_size=1 --output_dim=1 --seed %seed[8]% 
python -u train_baseline.py --mask_rate=0.6 --dataset_mode=cmumosi_multimodal  --model=mmin_AE --print_freq=10 --gpu_ids=0 --input_dim_a=250 --embd_size_a=128 --input_dim_v=1000 --embd_size_v=128 --input_dim_l=38400 --embd_size_l=128 --AE_layers=256,128 --ce_weight=1.0 --mse_weight=0.2 --cls_layers=128,128 --dropout_rate=0.5 --niter=20 --niter_decay=10 --init_type normal --batch_size=256 --lr=1e-3 --run_idx=8 --name=mmin  --batch_size=1 --output_dim=1 --seed %seed[9]% 



cd ..