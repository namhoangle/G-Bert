#! /bin/bash
echo seed="${seed}"
echo thres="${thres}"
python run_pretraining.py --model_name GBert-pretraining-thres-${thres} --num_train_epochs 5 --do_train --graph --seed ${seed}
python run_gbert.py --model_name GBert-predict-thres-${thres} --use_pretrain --pretrain_dir ../saved/GBert-pretraining-thres-${thres} --num_train_epochs 5 --do_train --graph --seed ${seed} --therhold ${thres}

for i in {1..15}
do
  echo "i=${i}"
  python run_pretraining.py --model_name GBert-pretraining-thres-${thres} --use_pretrain --pretrain_dir ../saved/GBert-predict-thres-${thres} --num_train_epochs 5 --do_train --graph --seed ${seed}
  python run_gbert.py --model_name GBert-predict-thres-${thres} --use_pretrain --pretrain_dir ../saved/GBert-pretraining-thres-${thres} --num_train_epochs 5 --do_train --graph --seed ${seed} --therhold ${thres}
done


