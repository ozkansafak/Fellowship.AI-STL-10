## METHODOLOGY

> * The labeled data set is used to train a `VGG16` model by finetuning. The last 3 dense layesrs were stripped and replaced with a new layer or layers and the training was done only on the added layer or layers. For this, `5,000` `STL-10` images were used to train and cross-validate on 5 folds.   


> * Once trained, the model is used to predict the classes of the unlabelled dataset. Then the softmax outputs of the unlabeled images were calculated via forward pass. If the model outputted a softmax probability that exceeded a threshold value (~0.9, a hyperparameter) on any class, then that image was labeled to that class.  


> * Once labelled, the images from the previously unlabelled data set were added to the training set. The network was trained **all at once** on the expanded data set.  The test accuracy was calculated on the `8,000 STL-10` images.



## NOTES
> * Cross validation data set was split off from the training dataset in k=5 folds.  The K-fold Cross-validaton accuracy differed from the Test Accuracy substantially (98.5 to 94%, respectively). This raised suspicions whether the test and training sets were pulled form the same distribution. I was compelled to mix all the labeled data set (13,000 images), shuffle and select 5,000 for Training and 8,000 for Test datasets. 


> * After `Unlabeled_X.bin` was read into array and resized to `(224,224,3)`. `X_unlabeled_features`, the bottleneck features were computed via forward pass on `model_bottom`. Then `Unlabeled_X.bin` was deleted to create space to write `X_unlabeled_features` array (9.5 GB) to the disk.  

> * The python code is run on an AWS p2.xlarge instance NVIDIA Tesla K80 Accelerators, that provides 12 GiB of GPU Memory.


## RESULTS

#### Supervised Results:
```
5-Fold Training + Cross-validation on 5,000 STL-10 images
Test on 8,000 STL-10 images


                                                                1000 relu Layer    
                                          1000 relu Layer       1000 relu Layer    
                     1000 Softmax Layer   1000 Softmax Layer    1000 Softmax Layer
----------------------------------------------------------------------------------
Avg Test Accuracy :  96.66%               94.36%                94.39%

```
#### Unsupervised Results:
```
Avg Test Accuracy
threshold = 0.99 :   95.24 %              N/A                   N/A 
threshold = 0.95 :   95.47 %              95.23 %               N/A 
threshold = 0.90 :   95.49 %              N/A                   N/A 
threshold = 0.80 :   95.49 %              N/A                   N/A

```

> * With the expansion of the original training dataset into the unlabeled dataset, the Avg Test Accuracy has improved by approximately **1 %** for the value **threshold = 0.95**.






## ALTERNATIVE METHODOLOGY & FUTURE WORK

> * An alternative method, would be to set the threshold to a high level at first (like 0.99). Then label the unlabelled set to add to and expand the training dataset. Once the model is trainined on a larger dataset, the model will be more confident and robust. Now, the unlabelled images, with softmax probabilities previously predicted to be less than 0.99, will output different softmax probabilities once trained on the updated model. the threshold, a hyperparameter, can be held constant or be lowered slowly to absorb more images from the unlabelled dataset into the expanded traning set.


