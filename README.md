# Conversion-of-Text-in-Image-to-Speech-using-Python
- Human communication today is mainly via speech and text. To access information in a text, a person needs to have vision. However those who are deprived of vision can gather information using their hearing capability.
- As reading is of prime importance in the daily routine (text being present everywhere from newspapers, commercial products, sign-boards, digital screens etc.) of mankind, visually impaired people face a lot of difficulties.
- This project can assist the visually impaired by reading out the text to them.
- It involves the design & development of a text in image to speech conversion using optical character recognition and text to speech technology

--------

 The first step is to take input in the form a **png, jpg, jpeg or jfif** file. Then it is fed into the **Laplace filter** which decides how ‘blurred’ the image is. If the score is below 50, the outputs will be very erroneous due to the blur present in the original image.
 
 The image preprocessing is applied where the dimensions of the image are changed to be compatible with the **EAST** [Efficient and Accurate Scene Text Detector] model neural network. 
 
 **EAST** model performs the text detection part and creates a bounding box for each of the word level text detected.
 This includes **Thresholding** and **NMS** [Non-Maximum Suppression] to output a single bounding box for the multiple boxes obtained based on their probability score.
 
 After the detection of text regions, the boxes are fed into the Tesseract **OCR** [Optical Character Recognition]. Before doing that, image preprocessing operations are applied to increase the accuracy of the text detected by Tesseract OCR.
 
 Finally, the output of the text is fed to **GTTS** [Google Text to Speech which is then saved to an mp3 file and played on command
