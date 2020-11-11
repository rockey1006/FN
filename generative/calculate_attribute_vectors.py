# MIT License
# 
# Copyright (c) 2017 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Calculate average latent variables (here called attribute vectors) 
for the different attributes in CelebA
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import argparse
import importlib
import facenet
import os
import numpy as np
import math
import time
import h5py
from six import iteritems
import facenet.src.facenet as facenet

def main(args):
  
    img_mean = np.array([134.10714722, 102.52040863, 87.15436554])
    img_stddev = np.sqrt(np.array([3941.30175781, 2856.94287109, 2519.35791016]))
    
    vae_checkpoint = os.path.expanduser(args.vae_checkpoint)#1.加载pretrained VAE
    
    fields, attribs_dict = read_annotations(args.annotations_filename)#2.加载attributes 的40个属性名称和每张图片的属性值
    
    vae_def = importlib.import_module(args.vae_def)#dfc-vae
    vae = vae_def.Vae(args.latent_var_size)#隐变量空间100
    gen_image_size = vae.get_image_size()

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        
        image_list = facenet.get_image_paths(os.path.expanduser(args.data_dir))
        
        # Get attributes for images
        nrof_attributes = len(fields)
        attribs_list = []
        for img in image_list:
            key = os.path.split(img)[1].split('.')[0]
            attr = attribs_dict[key]
            assert len(attr)==nrof_attributes
            attribs_list.append(attr)#每张图片的40维属性值 [1,-1...]
            
        # Create the input queue
        index_list = list(range(len(image_list)))
        input_queue = tf.train.slice_input_producer([image_list, attribs_list, index_list], num_epochs=1, shuffle=False)  #tensor list 包括图片，40维属性，图片总长度      
        
        nrof_preprocess_threads = 4
        image_per_thread = []
        for _ in range(nrof_preprocess_threads):
            filename = input_queue[0]#resize 图片
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, channels=3)
            image = tf.image.resize_image_with_crop_or_pad(image, 160, 160)
            #image = tf.image.resize_images(image, (64,64))
            image.set_shape((args.image_size, args.image_size, 3))
            attrib = input_queue[1]#attribs_list，
            attrib.set_shape((nrof_attributes,))
            image = tf.cast(image, tf.float32)
            image_per_thread.append([image, attrib, input_queue[2]])
    
        images, attribs, indices = tf.train.batch_join(
            image_per_thread, batch_size=args.batch_size, 
            shapes=[(args.image_size, args.image_size, 3), (nrof_attributes,), ()], enqueue_many=False,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        
        # Normalize
        images_norm = (images-img_mean) / img_stddev

        # Resize to appropriate size for the encoder 
        images_norm_resize = tf.image.resize_images(images_norm, (gen_image_size,gen_image_size))
        
        # Create encoder network
        mean, log_variance = vae.encoder(images_norm_resize, True)
        
        epsilon = tf.random_normal((tf.shape(mean)[0], args.latent_var_size))
        std = tf.exp(log_variance/2)
        latent_var = mean + epsilon * std##计算latent值     
        
        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        
        # Start running operations on the Graph
        gpu_memory_fraction = 1.0
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        

        with sess.as_default():
          
            if vae_checkpoint:
                print('Restoring VAE checkpoint: %s' % vae_checkpoint)
                saver.restore(sess, vae_checkpoint)
           
            nrof_images = len(image_list)
            nrof_batches = int(math.ceil(len(image_list) / args.batch_size))
            latent_vars = np.zeros((nrof_images, args.latent_var_size))
            attributes = np.zeros((nrof_images, nrof_attributes))
            for i in range(nrof_batches):
                start_time = time.time()
                latent_var_, attribs_, indices_ = sess.run([latent_var, attribs, indices])
                latent_vars[indices_,:] = latent_var_          
                attributes[indices_,:] = attribs_
                duration = time.time() - start_time
                print('Batch %d/%d: %.3f seconds' % (i+1, nrof_batches, duration))
            # NOTE: This will print the 'Out of range' warning if the last batch is not full,
            #  as described by https://github.com/tensorflow/tensorflow/issues/8330
             
            # Calculate average change in the latent variable when each attribute changes
            attribute_vectors = np.zeros((nrof_attributes, args.latent_var_size), np.float32)
            for i in range(nrof_attributes):#40
                pos_idx = np.argwhere(attributes[:,i]==1)[:,0]#在所有人的属性列表中：1.把某个属性==1的人记下 2.选择所有符合条件的人的值的集合比如40
                neg_idx = np.argwhere(attributes[:,i]==-1)[:,0]#同上
                pos_avg = np.mean(latent_vars[pos_idx,:], 0)#找到第40个隐变量的值，求100个值的平均值
                neg_avg = np.mean(latent_vars[neg_idx,:], 0)
                attribute_vectors[i,:] = pos_avg - neg_avg#计算 40个100维度的attribute_vectors：，某个属性的全部100个值==
            
            filename = os.path.expanduser(args.output_filename)
            print('Writing attribute vectors, latent variables and attributes to %s' % filename)
            fields = [n.encode("ascii", "ignore") for n in fields]
            mdict = {'latent_vars':latent_vars, 'attributes':attributes, 
                     'fields':fields, 'attribute_vectors':attribute_vectors }
            with h5py.File(filename, 'w') as f:
                for key, value in iteritems(mdict):
                  
                  f.create_dataset(key, data=value)
                    
                    
def read_annotations(filename):
    attribs = {}    
    with open(filename, 'r') as f:
      
      for i, line in enumerate(f.readlines()):
        if i==0:
          continue  # First line is the number of entries in the file
        elif i==1:
            fields = line.strip().split() # Second line is the field names
        else:
              line = line.split()
              img_name = line[0].split('.')[0]
              img_attribs = list(map(int, line[1:]))
              attribs[img_name] = img_attribs
      return fields, attribs

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('vae_def', type=str,
        help='Model definition for the variational autoencoder. Points to a module containing the definition.', 
        default='src.generative.models.dfc_vae')
    parser.add_argument('vae_checkpoint', type=str,
        help='Checkpoint file of a pre-trained variational autoencoder.')
    parser.add_argument('data_dir', type=str,
        help='Path to the directory containing aligned face patches for the CelebA dataset.')
    parser.add_argument('--annotations_filename', type=str,
        help='Path to the annotations file',
        default='/media/deep/datasets/CelebA/Anno/list_attr_celeba.txt')
    parser.add_argument('--output_filename', type=str,
        help='Filename to use for the file containing the attribute vectors.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=128)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=64)
    parser.add_argument('--latent_var_size', type=int,
        help='Dimensionality of the latent variable.', default=100)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)

    return parser.parse_args(argv)
  
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
