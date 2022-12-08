#! /bin/bash
echo seed="${seed}"
python run_pretraining.py --model_name GBert-pretraining-q-${seed} --num_train_epochs 5 --do_train --graph --seed ${seed}
python run_gbert.py --model_name GBert-predict-q-${seed} --use_pretrain --pretrain_dir ../saved/GBert-pretraining-q-${seed} --num_train_epochs 5 --do_train --graph --seed ${seed}

for i in {1..15}
do
  echo "i=${i}"
  python run_pretraining.py --model_name GBert-pretraining-q-${seed} --use_pretrain --pretrain_dir ../saved/GBert-predict-q-${seed} --num_train_epochs 5 --do_train --graph --seed ${seed}
  python run_gbert.py --model_name GBert-predict-q-${seed} --use_pretrain --pretrain_dir ../saved/GBert-pretraining-q-${seed} --num_train_epochs 5 --do_train --graph --seed ${seed}
done


