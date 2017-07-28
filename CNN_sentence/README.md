# My Version of CNN Sentence Classification in Theano

### Data Preprocessing
To process the raw data, run

```
python process_data.py path
```

### Using the GPU
```
 THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cer_main.py
```

### Explain
My refactoring code is named cer_*, which is 
- **cer_main.py** : entrance of the code.
- **cer_model.py** : class CNN_Sen_Model, the whole neural network.
- **cer_module.py** ； detailed part of the network.

**model.json** is the config file of the model, which made it easier to change the model.