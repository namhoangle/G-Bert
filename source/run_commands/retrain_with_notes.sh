#! /bin/bash
python run_pretraining.py --model_name GBertNotes-pretraining-seed$1 --num_train_epochs 5 --do_train --graph --seed $1
python run_gbert_notes.py --model_name GBertNotes-predict-seed$1 --use_pretrain --pretrain_dir ../saved/GBertNotes-pretraining-seed$1 --num_train_epochs 5 --do_train --graph --seed $1

for i in {1..15}
do
python run_pretraining.py --model_name GBertNotes-pretraining-seed$1 --use_pretrain --pretrain_dir ../saved/GBertNotes-predict-seed$1 --num_train_epochs 5 --do_train --graph --seed $1
python run_gbert_notes.py --model_name GBertNotes-predict-seed$1 --use_pretrain --pretrain_dir ../saved/GBertNotes-pretraining-seed$1 --num_train_epochs 5 --do_train --graph --seed $1
done


