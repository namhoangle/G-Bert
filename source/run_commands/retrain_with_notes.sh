#! /bin/bash
# python run_pretraining.py --model_name GBertNotes-pretraining-q --num_train_epochs 5 --do_train --graph --seed $1
# python run_gbert.py --model_name GBertNotes-predict-q --use_pretrain --pretrain_dir ../saved/GBertNotes-pretraining-q --num_train_epochs 5 --do_train --graph --seed $1

for i in {1..15}
do
python run_pretraining.py --model_name GBertNotes-pretraining-q --use_pretrain --pretrain_dir ../saved/GBertNotes-predict-q --num_train_epochs 5 --do_train --graph --seed $1
python run_gbert.py --model_name GBertNotes-predict-q --use_pretrain --pretrain_dir ../saved/GBertNotes-pretraining-q --num_train_epochs 5 --do_train --graph --seed $1
done


