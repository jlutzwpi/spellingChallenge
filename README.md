# Spelling Challenge
Python spelling challenge game for kids using computer vision.  A picture comes up with a letter missing from the word.  Since this is for my 5 year old, it's always the first letter, but I wrote the code so you can change the letter location.  Once you write in the letter, if it matches, the word is said in spanish and printed on the screen as well!

![alt text](https://github.com/jlutzwpi/spellingChallenge/blob/main/screenshots.png?raw=true)

This was developed on a Jetson Nano. I started on the Raspberry Pi 4 (and ported over).
Requires OpenCV and Tensorflow.  I created the model using the KaggleAZ dataset and Google Colab.
My configuration is: 
1. Jetpack 4.6 
2. TensorRT 8.0
3. OpenCV 4.5.
4. CUDA 10.2.


