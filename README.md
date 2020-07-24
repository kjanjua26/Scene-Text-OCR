# Scene-Text-OCR
This repository contains code for Scene Text OCR following the famous CRNN model. The detection part is handled by EAST and the recognition is done by CRNN.

# Detection

For the detection part, I use EAST detector => <a href="https://github.com/argman/EAST">CODE HERE</a>. I finetune the network on products data.

## Detection Results

<p float="left">
  <img src="/imgs/4.jpg" height="300" width="200" />
  <img src="/imgs/5.jpg" height="300" width="200" /> 
  <img src="/imgs/6.jpg" height="300" width="200" />
</p>

# Recognition

For the recognition part, I have implemented the CRNN network in tensorflow and finetuned it on the data. The data can be downloaded from here => <a href="https://drive.google.com/file/d/1NPF1OSsaak7oUr7Hz-w9mZj0ePLzqqPz/view?usp=sharing">DOWNLOAD HERE</a>
