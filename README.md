# CEDL_FINAL Face Completion
## Dataset
* Use [**CELEBA**](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
* Generate noise in center (64x64) in utils.py
## Autoencoder  
### Run    
* Load pre-trained  
Put [checkpoint files](https://drive.google.com/drive/folders/1aFRcOunF2WOcjL0nBdBYtAWs0u_ksUsr?usp=sharing) under ./autoencoder/model folder and set restore=True 
```
python main.py
--epoch 
--batch_size
--data_path
--model_path
--output_path
--graph_path
--restore
--mode=train/test
```
### Result
<img src='./Readmefile/in_ag.png' width = "200" height = "200"> <img src='./Readmefile/out_a.png' width = "200" height = "200">

## Autoencoder + GAN
### Run
```
python main.py
--epoch 
--batch_size
--data_path
--model_path
--output_path
--graph_path
--restore
--mode=train/test
```
### Network
<img src='./Readmefile/autoencoder_gan_arch.png'>

### Loss
<img src='./Readmefile/loss.png'>

### Result
<img src='./Readmefile/in_ag.png' width = "200" height = "200"> <img src='./Readmefile/out_ag.png' width = "200" height = "200">
