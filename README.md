# Streetview_synthesize

# Pipeline
1. Prepare Cityscapes human dataset
2. Prepare Inpainting dataset
3. Inpainting
4. Produce CityPedestrian dataset
5. Produce Non-existing pedestrian dataset
6. Non-existing pedestrian detection

## Cityscapes human dataset
which contains many pedestrians in a single view
- training: 194 [ from 2975]
- valid: 37 [ from 500 ]
- size: 256 x 512
- mask had been diliated
  ###Process
  1. split images by human ration [human ratio >= 0.05]
  2. images resize
  3. mask dilation

## Inpainting dataset
which contains no pedestrian
- traing: 13139 [select from 2975 + 19998, human ratio = 0.00]
- mask: 194 [ from 2975, substitute of random mask]
- test: Cityscapes human dataset [training, valid]

## Inpainting
remove the pedestrian in human dataset
1. train Context encoders feature learning 
2. [generated + real] + poisson blending -> inpainting_context

## CityPedestrian dataset
split each pedestrian in a single view
- training: 4175 [from 194]
- valid: 811 [from 37]
    ###Process
    1. split pedestrians for pose estimation [create_instance_for_pos]
    2. re permuate estimation result [re_permuate_h5]
    3. porduce datasetPed [create_datasetPed_cityscape]

## Non-existing pedestrian dataset
- training: 6509 [from 194, context + style]
- valid: [from 37, context + style]
  ### Process
  1. create different combinations
  2. split pedestrians into {exist, non-exist}
  3. put back to {image, heatmp} respectively
  4. [create_dataset_heatmap_pos_coordinate]
  5. resize to [128, 512]

## Non-existing pedestrian detection
1. FCN [FCN_haetmap_instance_pose, batch_size: 9]
