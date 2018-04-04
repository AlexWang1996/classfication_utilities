# tfrecords_datasets_split


These scrips help users split the trainning and testing(or called validation) datasets in TensorFlow or Keras
____

## TensorFlow
* In TensorFlow, one common way to efficiently use data and datasets is to stored them in TFRecord files.
* When used, I access them throught batches and send them to our train or test model.

## Keras
* In Keras, it support more simply ways to generate our datasets.
* Typcially, here I use the flow_from_directory method to prepare training and validation datasets.

### About the path
* Here I specify the path I used to generate TFRecord files and Keras's training and validation datasets
  > /user     
  >> /image_data     
  >>> /dog      
  >>> /cat      
  >>> /elephant     
  >>> ...     
  >>> /tiger      




