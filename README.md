# Byte Size ASL

## Inspiration
Despite modern technological innovation, the advent and value of artificial intelligence has largely gone unused in the domain of education. Byte Size ASL aims to encourage the integration of computational thinking within the very foundation of contemporary society. **American Sign Language (ASL)** is used only by approximately **0.2%** of the American population. Byte Size ASL seeks to prove the value of machine learning in education by providing an accessible methodology as a means to deliver a gateway into the world of ASL—_byte sized_.

## What it does and how to use it
To make effective use of Byte Size ASL:
• Position your hand in the center of the webcam viewport.
• Ensure that your hand is well illuminated and that the background is clear.
• Above the webcam-viewport you will see the generated spelling-word. **The objective of Byte Size ASL is to recreate this spelling-word using the ASL alphabet.**
• Press the spacebar to take a picture of your hand. Byte Size ASL's state of the art convolutional neural network will evaluate this image based on 92,000 unique features to   determine which sign you are forming.
• If your input is correct, the letter will be added to the progress section, until you have completed the spelling-word.

## How we built it
The convolutional neural network that powers Byte Size ASL was built using the Tensorflow and Keras libraries. To train the model, a data set containing 28x28 pixel gray-scale images was used. Data preparation involved one-hot encoding the labels, normalizing pixel values, and adding a color channel to pass the data through the convolutional layers.

The model architecture consists of several convolutional layers followed by max pooling layers. After going through the convolutional layers, the data is then flattened into a one-dimensional vector so as to pass through three densely connected layers. The 'softmax' activation function was used for the last dense layer. A series of dropout layers were also added to reduce over-fitting. The 'categorical_crossentropy' loss function was used to train the model.

The front-end application was constructed using HMTL5 and CSS3, with embedded JavaScript to both render the webcam stream entirely clientside, and to POST data to to the server. At the server endpoint, we used Flask to handle GET requests and deliver the results of the machine learning model.

Dataset Source: https://www.kaggle.com/datamunge/sign-language-mnist

## Challenges we ran into
One of the most pressing challenges for us was synchronizing the front-end application with our server-side architecture. Additionally, image classification with machine learning is **very** difficult and required a very specific set of criteria to be met in order to accurately classify hand signs. The model was trained on grayscale images, therefore, some of the real-world samples were difficult to precisely process. Consequently, the model does not perform well in medium or dim lighting conditions. Furthermore, to save training time, the images were scaled down immensely at the cost of prediction accuracy.

## Accomplishments that we're proud of
We are very proud of the performance of our neural network classifier on our testing set, netting a validation accuracy of 98%. We are also proud of the responsive design of the frontend application.

## What we learned
One of the most valuable takeaways from this project was the in-depth approach we took into server architecture. We delved into many new concepts and technologies, (new for us) including AJAX, JavaScript, and the Python Flask microservice framework. We learned the useful applications as well as the constraints of utilizing machine learning to deliver education effectively.

## What's next for Byte Size ASL
While we are more than satisfied with our Machine Learning engineer's capability to produce a 98% validation accuracy, we recognize that the neural network has trouble in predicting many single sample inputs. We aim to improve the predictive accuracy of our neural network, using pratical and higher quality imaging to train the model, so that we can more accurately deliver a machine learning experience.
