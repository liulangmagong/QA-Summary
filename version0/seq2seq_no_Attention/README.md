# 模型架构
```Python
def seq2seq()
    model = Sequential()
    
    model.add(Embedding(...))
    model.add(Bidirectional(GRU(...)))
    model.add(Dense(...)) 
    model.add(RepeatVector(...))
    
    model.compile(loss=..., optimizer=...)
    model.summary()
    
    return model

model = seq2seq()
model.fit()

# 模型保存
model.save('model_save.h5')
del model
model = tf.keras.models.load_model('model_save.h5')
```

