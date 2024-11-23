import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,LayerNormalization,MultiHeadAttention,Input,Dropout,concatenate
from keras.optimizers import Adam




class VIT:
    def __init__(self,patch_size=16,dropout_rate=0.2,num_classes=10,num_heads=2):
        """
            Vision Transformer(ViT) implementation in TensorFlow keras.

            args:
                pathch_size(int) : size of patch extract from image-default(16)
                dropout_rate(float) : Dropout regularization-default(0.2)
                num_classes(int) = classification classes 
                num_heads(int) = attention head-default(2)

        """
        self.patch_size = patch_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.num_heads = num_heads

    def patch_embed(self,image):
        """
            patch extraction from the image
        args:
            image : input must be tensor 

        return:
            image patchs $

        """
        
        self.image = image
        if np.ndim(self.image) == 3:
            self.image = tf.expand_dims(self.image,axis=0)
        
        
        patch = tf.image.extract_patches(images=self.image,sizes=[1,self.patch_size,self.patch_size,1]\
                                         ,strides=[1,self.patch_size,self.patch_size,1],rates=[1,1,1,1],padding='VALID')
        
        self.patchsize = tf.shape(patch)[0]
        self.num_patchs = tf.shape(patch)[1]*tf.shape(patch)[2]
        self.depth = tf.shape(patch)[-1]
        patchs = tf.reshape(patch,(self.patchsize,self.num_patchs,self.depth))

        return patchs

    def positional_empedding(self,patches):
        """
            add positional empedding to the patchs & classification empedding

        args:
            patchs : after patch empedding the extracted img patchs [tensor]

        return :
            tensor with positional and classification empedded


        """
        
        linear_projection= Dense(int(self.depth))(patches) # linear empedding
     
        cls_token = tf.Variable(tf.zeros((self.patchsize,1,self.depth)),trainable=True)
        cls_empedding = tf.concat([cls_token,linear_projection],axis=1) #cls empedding
     
        position_emp = tf.Variable(tf.random.normal([1,self.num_patchs+1,self.depth]),trainable=True)
        cls_empedding += position_emp

        return cls_empedding


    def transformer_encoder(self,input):
        """
            Transformer encoder arch

        args:
            input : input patchs [tensor]

        return:
            output: tensor with same shape as input 




        """

        if np.ndim(input) == 3:
            input = tf.expand_dims(input,axis=0)

        shape = np.shape(input)
        norm = LayerNormalization(epsilon=0.0001)(input)
        attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=64//self.num_heads)(norm,norm)
        attention = Dropout(self.dropout_rate)(attention)

        residual = attention+input

        norm = LayerNormalization(epsilon=0.0001)(residual)
        mlp = Dense(shape[-1]*2,activation='relu')(norm)
        mlp = Dropout(self.dropout_rate)(mlp)
        mlp = Dense(shape[-1],activation='relu')(mlp)
        mlp = Dropout(self.dropout_rate)(mlp)
        mlp = Dense(shape[-1])(mlp)
        residual1 = residual+mlp

        outputs = LayerNormalization(epsilon=1e-6)(residual1)

        return outputs

    def build_vit(self,image,transformer_depth=5):
        """
            build vit
        args:
            transformer_depth(int) = depth of the transormer encoder layer 
            image : tensor
        """
        print('s')
        patch_process = self.patch_embed(image)
        print('s')
        x = self.positional_empedding(patch_process)
        print('s')
        for i in range(transformer_depth):
            x = self.transformer_encoder(x)
            print('s')



        cls_token = x[:, 0, :]
        cls = Dense(np.shape(cls_token)[-1])(cls_token)
        cls = Dense(100,activation='relu')(cls)
        cls = Dense(self.num_classes,activation='softmax')(cls)



    def train_vit(self, x_data, y_data, transformer_depth=6, epochs=10, batch_size=32, learning_rate=1e-4):
        """
            vit training

        args:
            x_data (np.ndarray): Training images
            y_data (np.ndarray): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for the optimizer
            transformer_depth (int): Number of transformer encoder layers
             
        """

        model = self.build_vit(transformer_depth,image)

        model.compile(optimizer=Adam(learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])

        model.fit(x_data,y_data,epoch=epochs,batch_size=batch_size)









