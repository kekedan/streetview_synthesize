# Streetview Synthesize
Nonexistent Pedestrian Detection
- [CVPR2017 workshop poster](https://drive.google.com/open?id=0B4AGJgH-EFvIYUtqb1M4dTlHY3M)
- [intro video](https://www.youtube.com/watch?v=t8IHB29uTQM)
- [intro power point](https://drive.google.com/open?id=0B4AGJgH-EFvIUEdySV9FQ2hGVlE)
- [ICCV2017 workshop]


# Pipeline
1. Prepare Cityscapes human dataset
2. Prepare Inpainting dataset
3. Inpainting
4. Produce CityPedestrian dataset
5. Produce Non-existing pedestrian dataset
6. Non-existing pedestrian detection
7. Synthesize new image(currently version: copy-and-paste)

## Cityscapes Human Dataset
which contains many pedestrians in a single street view
- training: 194 [ from 2975]
- valid: 37 [ from 500 ]
- size: 256 x 512
- mask had been diliated
  ###Process
  1. split images by human ratio [human ratio >= 0.05]
  2. images resize
  3. mask dilation

## Inpainting Dataset
which contains no pedestrian
- training: 13139 [select from 2975 + 19998, human ratio = 0.00]
- mask: 194 [ from 2975, substitute of random mask]
- test: Cityscapes human dataset [training, valid]

## Inpainting
remove the pedestrians in human dataset
1. train Context Encoders Feature Learning 
2. [generated + real] + poisson blending -> inpainting_context
3. High resolution(improved)

## CityPedestrian Dataset
split each pedestrian in a single view
- training: 4175 [from 194]
- valid: 811 [from 37]
    ###Process
    1. split pedestrians for pose estimation [create_instance_for_pos]
    2. re permuate estimation result [re_permuate_h5]
    3. produce datasetPed [create_datasetPed_cityscape]

## Non-existing Pedestrian Dataset
- training: 6509 [from 194, context + style]
- valid: [from 37, context + style]
  ### Process
  1. create different combinations
  2. split pedestrians into {exist, non-exist}
  3. put back to {image, heatmp} respectively
  4. [create_dataset_heatmap_pos_coordinate]
  5. resize to [128, 256]

## Non-existing Pedestrian Detection
1. FCN [FCN_haetmap_instance_pose, batch_size: 9]
2. Stacked Hourglass
3. FCN+D

## Folder
- Context Encoders_Feature Learning by Inpainting: image inpainting
- DCGAN: some experiments about detecting non-existing pedestrians with DCGAN-like model
- FCN: some experiments about detecting non-existing pedestrians with FCN model
- 