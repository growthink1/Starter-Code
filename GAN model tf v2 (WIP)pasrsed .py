#!/usr/bin/env python
# coding: utf-8

# ## GAN Model Code in tf V2. (Runnable but not optmized for V2)

# In[39]:


import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd 
import os 
import time 


# In[40]:


class DCGAN_v2():
    
    
    def __init__(
            self,
            batch_size=100,
            buffer_size= 1000,
            epochs = 100, 
            image_shape=[24,24,1], #[1024,1024,1],
            dim_z=24,
            dim_y=1,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_channel=1
            ):       
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epochs = epochs
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel
           

    def build_discriminator_model(self):
        Y = keras.Input(shape=self.dim_y)
        image = keras.Input(shape = self.image_shape)
        yb = layers.Reshape([1, 1, self.dim_y]) (Y)
        X = tf.concat([image, yb * tf.ones([24, 24, self.dim_y])],3)
        h1 = layers.LeakyReLU(alpha=0.2)(layers.Conv2D(filters=self.dim_W3,kernel_size=5,strides=(2,2), padding='same',data_format="channels_last") (X))
        h1 = tf.concat([h1, yb * tf.ones([12, 12, self.dim_y])],3)
        h2 = layers.LeakyReLU(alpha=0.2)(layers.BatchNormalization()(layers.Conv2D(filters=self.dim_W2,kernel_size=5,strides=(2,2), padding='same',data_format="channels_last") (h1)))                                                                                                                                                                
        h2 = layers.Flatten()(h2)
        h2 = tf.concat([h2, Y], 1)
        h3 = layers.LeakyReLU(alpha=0.2)(layers.BatchNormalization()(layers.Dense(self.dim_W1)(h2)))
        model = keras.Model(inputs=[image,Y], outputs=h3, name="discriminator_model")
        return model 
    
    def build_generator_model(self):
        Y = keras.Input(shape=self.dim_y)
        Z = keras.Input(shape=self.dim_z)

        yb = layers.Reshape([1, 1, self.dim_y]) (Y)
        A = tf.concat([Z,Y],1)
        h1=layers.ReLU()(layers.BatchNormalization()(layers.Dense(self.dim_W1)(A)))
        h1 = tf.concat([h1, Y],1)
        h2 =layers.ReLU()(layers.BatchNormalization()(layers.Dense(self.dim_W2*6*6)(h1)))
        h2 =layers.Reshape([6, 6, self.dim_W2]) (h2)
        h2 = tf.concat([h2, yb*tf.ones([6,6, self.dim_y])],3)
        h3 =layers.ReLU()(layers.BatchNormalization() (layers.Conv2DTranspose(filters = self.dim_W3 , kernel_size = 5, strides = (2,2), padding = "same", data_format="channels_last")(h2)))
        h3 = tf.concat([h3, yb*tf.ones([12, 12, self.dim_y])], 3)
        h4 = layers.Conv2DTranspose(filters = self.dim_channel , kernel_size = 5, strides = (2,2), padding = "same", data_format="channels_last")(h3)       
        model = keras.Model(inputs=[Z,Y], outputs=h4,name="generator_model")
        return model
    
    def discriminator_loss(self, real_output, gen_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        gen_loss = cross_entropy(tf.zeros_like(gen_output), gen_output)
        total_loss = real_loss + gen_loss
        return total_loss
    
    def generator_loss(self, gen_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(gen_output), gen_output)
    
    
    ### TESTING IN PROGRESSS ###
    # @tf.function
    def train_step(self, Z, Y, image_real):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # generator output 
            h4 = generator([Z,Y], training=True)
            image_gen = tf.keras.activations.sigmoid(h4)
            
            image_real = layers.LeakyReLU()(layers.BatchNormalization() (layers.Dense(24*24, use_bias=False)(image_real)))
            image_real = layers.Reshape(self.image_shape)(image_real) 
            
            real_output = discriminator([image_real,Y], training=True)
            gen_output = discriminator([image_gen,Y], training=True)

            # compute losss
            gen_loss = self.generator_loss(gen_output)
            disc_loss = self.discriminator_loss(real_output, gen_output) 

        # compute gradients 
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # update parameters 
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        
    def train(self, Z, Y, image_real, checkpoint_dir = './training_checkpoints', save_frequency = 1):
    
        # put data into batches 
        assert (Z.shape[0] == Y.shape[0] == image_real.shape[0])   
        num_training_data= Z.shape[0]
        num_features = Z.shape[1]
        training_dataset = tf.concat((Z,Y,image_real), axis = 1)
        train_dataset = tf.data.Dataset.from_tensor_slices(training_dataset).shuffle(self.buffer_size).batch(self.batch_size)

        global generator_optimizer 
        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        global discriminator_optimizer 
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        global generator 
        generator = self.build_generator_model()
        global discriminator 
        discriminator = self.build_discriminator_model()


        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

        for epoch in range(self.epochs):
            start = time.time()
            
            batch_number = 1
            
            for train_batch in train_dataset:
                Z_batch = train_batch[:, :num_features]
                Y_batch = train_batch[:, num_features]
                image_real_batch = train_batch[:, num_features+1:]
                self.train_step(Z_batch, Y_batch, image_real_batch)
                # Save the model every epochs
                if (epoch + 1) % save_frequency == 0:
                    checkpoint.save(file_prefix = checkpoint_prefix)

                print ('Time for epoch {} batch {} is {} sec'.format(epoch + 1, batch_number, time.time()-start))
                batch_number +=1

        checkpoint.save(file_prefix = checkpoint_prefix)

    


# In[41]:


num_training_data = 400
num_feature = 24

Z_train = tf.random.normal((num_training_data,num_feature))
Y_train = tf.random.normal((num_training_data, 1))
image_real_train = tf.random.normal((num_training_data,num_feature))

dcgan = DCGAN_v2(epochs=2)


# In[42]:


dcgan.train(Z_train, Y_train, image_real_train)


# In[43]:


dcgan = DCGAN_v2()
generator_model = dcgan.build_generator_model()
generator_model.summary()


# In[44]:


discriminator_model = dcgan.build_discriminator_model()
discriminator_model.summary()


# In[ ]:





# ## ref = https://www.tensorflow.org/tutorials/generative/dcgan
# ## ref 2 = https://www.tensorflow.org/guide/keras/functional
# ## ref 3 = https://www.tensorflow.org/tutorials/images/cnn

# ## SCRATCH

# In[84]:


h4 = self.generate(Z, Y)
#image_gen comes from sigmoid output of generator
image_gen = tf.nn.sigmoid(h4)

raw_real2 = self.discriminate(image_real, Y)
#p_real = tf.nn.sigmoid(raw_real)
p_real = tf.reduce_mean(raw_real2)

raw_gen2 = self.discriminate(image_gen, Y)
#p_gen = tf.nn.sigmoid(raw_gen)
p_gen = tf.reduce_mean(raw_gen2)


discrim_cost = tf.reduce_mean(raw_real2) - tf.reduce_mean(raw_gen2)
gen_cost = -tf.reduce_mean(raw_gen2)

# tensor that changes input shape  continuously 
# filters, just using different sizes of filters to prevent bias from happening 


contextual_loss_latter = tf.keras.layers.Flatten()(
    -tf.math.log(
        (mask + tf.math.multiply(tf.ones_like(mask) - mask, pred_high)) - tf.math.multiply(
            tf.ones_like(mask) - mask, image_gen))
    - tf.math.log(
        (mask + tf.math.multiply(tf.ones_like(mask) - mask, image_gen)) - tf.math.multiply(
            tf.ones_like(mask) - mask, pred_low)))
contextual_loss_latter = tf.where(tf.math.is_nan(contextual_loss_latter), tf.ones_like(contextual_loss_latter) * 1000000.0, contextual_loss_latter)
contextual_loss_latter2 = tf.math.reduce_sum(contextual_loss_latter, 1)
#square loss

contextual_loss_former = tf.math.reduce_sum(tf.keras.layers.Flatten()(
    tf.math.square(tf.math.multiply(mask, image_gen) - tf.math.multiply(mask, image_real))), 1)
contextual_loss_prepare = tf.math.reduce_sum(tf.keras.layers.Flatten()(
    tf.math.square(tf.math.multiply(tf.ones_like(mask) - mask, image_gen) - tf.math.multiply(tf.ones_like(mask)-mask, image_real))), 1)
perceptual_loss = gen_cost

complete_loss = contextual_loss_former + self.lam * perceptual_loss + 0.05*contextual_loss_latter2

grad_complete_loss = tf.GradientTape(complete_loss, Z)

grad_uniform_loss = tf.GradientTape(contextual_loss_prepare, Z)


# In[114]:


("image_real,", "image_gen,", "mask", "=", "None", ",", "pred_high", "=", "None,", "pred_low", "=", "None):")
if mask is None:
    mask = tf.random.normal([self.batch_size] + self.image_shape)
if pred_high is None:
    pred_high = tf.random.normal([self.batch_size]+self.image_shape)
if pred_low is None: 
    pred_low = tf.random.normal([self.batch_size]+self.image_shape)

discrim_cost = tf.reduce_mean(real_output) - tf.reduce_mean(gen_output)
gen_cost = -tf.reduce_mean(gen_output)

# contextual_loss_latter
contextual_loss_latter = layers.Flatten()(
        -tf.math.log(
            (mask + tf.math.multiply(tf.ones_like(mask) - mask, pred_high)) - tf.math.multiply(
                tf.ones_like(mask) - mask, image_gen))
        - tf.math.log(
            (mask + tf.math.multiply(tf.ones_like(mask) - mask, image_gen)) - tf.math.multiply(
                tf.ones_like(mask) - mask, pred_low)))
contextual_loss_latter = tf.where(tf.math.is_nan(contextual_loss_latter), tf.ones_like(contextual_loss_latter) * 1000000.0, contextual_loss_latter)
contextual_loss_latter2 = tf.math.reduce_sum(contextual_loss_latter, 1)

# contextual loss former
contextual_loss_former = tf.math.reduce_sum(tf.keras.layers.Flatten()(
        tf.math.square(tf.math.multiply(mask, image_gen) - tf.math.multiply(mask, image_real))), 1)

contextual_loss_prepare = tf.math.reduce_sum(tf.keras.layers.Flatten()(
        tf.math.square(tf.math.multiply(tf.ones_like(mask) - mask, image_gen) - tf.math.multiply(tf.ones_like(mask)-mask, image_real))), 1)
    

complete_loss = contextual_loss_former + self.lam * perceptual_loss + 0.05*contextual_loss_latter2
return complete_loss 


# In[ ]:




