# covid-xray

## Train

You just need to have the standard PyTorch and python libraries to run the training process.

Unzip the CoronaHack and Radiography in the root directory of the repository. Then if you want to add the shuffled images then just run the shuffledImages script.

You can simply run the train.py script to train the model. The model will be trained on the Radiography dataset and will be tested on the CoronaHack dataset.

train2.py trains the model on the CoronaHack dataset and tests using the Radiography dataset.

Other files in the repository are just helper scripts used to create data loaders and other functions.


## License

[MIT](https://choosealicense.com/licenses/mit/)
