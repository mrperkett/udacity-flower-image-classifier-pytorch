#! /usr/bin/env bash

flower_data_dir="/mnt/c/Large Files/flower_data"
category_names_file="cat_to_name.json"
learning_rate=0.001
num_hidden_units=512
dropout=0.2
num_epochs=5
gpu_flag="--gpu"
checkpoint_base="checkpoint-nhidden_${num_hidden_units}-nepochs_${num_epochs}"





for model_name in "alexnet" "densenet121" "resnet18" "vgg13" "vgg16"
do
    echo -e "\n####################################"
    echo "${model_name}"
    echo "####################################"

    # train model
    python3 ./train.py "${flower_data_dir}" --save_dir "."  --arch ${model_name} --learning_rate ${learning_rate} --hidden_units ${num_hidden_units} --epochs ${num_epochs} ${gpu_flag} --category_names "${category_names_file}" --dropout ${dropout}

    # rename checkpoint.pt
    checkpoint_file="${checkpoint_base}-${model_name}.pt"
    mv checkpoint.pt "${checkpoint_file}"

    # run predictions on test images
    for image_file in "test_images/cautleya_spicata.jpg" "test_images/hard-leaved_pocket_orchid.jpg" "test_images/orange_dahlia.jpg" "test_images/wild_pansy.jpg"
    do
        echo "Predictions for ${image_file}"
        python3 ./predict.py "${image_file}" "${checkpoint_file}" --top_k 5 --category_names "${category_names_file}" ${gpu_flag}
    done 
done


