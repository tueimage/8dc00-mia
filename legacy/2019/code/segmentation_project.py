#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project code+scripts for 8DC00 course
"""

# Imports

import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import segmentation as seg


def segmentation_mymethod(train_data_matrix, train_labels_matrix, test_data, task='brain'):
    # segments the image based on your own method!
    # Input:
    # train_data_matrix   num_pixels x num_features x num_subjects matrix of
    # features
    # train_labels_matrix num_pixels x num_subjects matrix of labels
    # test_data           num_pixels x num_features test data
    # task           String corresponding to the segmentation task: either 'brain' or 'tissue'
    # Output:
    # predicted_labels    Predicted labels for the test slice

    #------------------------------------------------------------------#
    #TODO: Implement your method here
    #------------------------------------------------------------------#
    return predicted_labels


def segmentation_demo():

    train_subject = 1
    test_subject = 2
    train_slice = 1
    test_slice = 1
    task = 'brain'

    #Load data
    train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
    test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)

    predicted_labels = seg.segmentation_atlas(None, train_labels, None)

    err = util.classification_error(test_labels, predicted_labels)
    dice = util.dice_overlap(test_labels, predicted_labels)

    #Display results
    true_mask = test_labels.reshape(240, 240)
    predicted_mask = predicted_labels.reshape(240, 240)

    # fig = plt.figure(figsize=(8,8))
    # ax1 = fig.add_subplot(111)
    # ax1.imshow(true_mask, 'gray')
    # ax1.imshow(predicted_mask, 'viridis', alpha=0.5)
    # print('Subject {}, slice {}.\nErr {}, dice {}'.format(test_subject, test_slice, err, dice))

    ## Compare methods
    num_images = 5
    num_methods = 3
    im_size = [240, 240]

    all_errors = np.empty([num_images,num_methods])
    all_errors[:] = np.nan
    all_dice = np.empty([num_images,num_methods])
    all_dice[:] = np.nan

    all_subjects = np.arange(num_images)
    train_slice = 1
    task = 'brain'
    all_data_matrix = np.empty([train_data.shape[0],train_data.shape[1],num_images])
    all_labels_matrix = np.empty([train_labels.size,num_images], dtype=bool)

    #Load datasets once
    print('Loading data for ' + str(num_images) + ' subjects...')

    for i in all_subjects:
        sub = i+1
        train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
        all_data_matrix[:,:,i] = train_data
        all_labels_matrix[:,i] = train_labels.flatten()

    print('Finished loading data.\nStarting segmentation...')

    #Go through each subject, taking i-th subject as the test
    for i in np.arange(num_images):
        sub = i+1
        #Define training subjects as all, except the test subject
        train_subjects = all_subjects.copy()
        train_subjects = np.delete(train_subjects, i)

        train_data_matrix = all_data_matrix[:,:,train_subjects]
        train_labels_matrix = all_labels_matrix[:,train_subjects]
        test_data = all_data_matrix[:,:,i]
        test_labels = all_labels_matrix[:,i]
        test_shape_1 = test_labels.reshape(im_size[0],im_size[1])

        fig = plt.figure(figsize=(15,5))

        predicted_labels = seg.segmentation_combined_atlas(train_labels_matrix)
        all_errors[i,0] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,0] = util.dice_overlap(test_labels, predicted_labels)
        predicted_mask_1 = predicted_labels.reshape(im_size[0],im_size[1])
        ax1 = fig.add_subplot(131)
        ax1.imshow(test_shape_1, 'gray')
        ax1.imshow(predicted_mask_1, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,0], all_dice[i,0])
        ax1.set_xlabel(text_str)
        ax1.set_title('Subject {}: Combined atlas'.format(sub))

        predicted_labels = seg.segmentation_combined_knn(train_data_matrix,train_labels_matrix,test_data)
        all_errors[i,1] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,1] = util.dice_overlap(test_labels, predicted_labels)
        predicted_mask_2 = predicted_labels.reshape(im_size[0],im_size[1])
        ax2 = fig.add_subplot(132)
        ax2.imshow(test_shape_1, 'gray')
        ax2.imshow(predicted_mask_2, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,1], all_dice[i,1])
        ax2.set_xlabel(text_str)
        ax2.set_title('Subject {}: Combined k-NN'.format(sub))

        predicted_labels = segmentation_mymethod(train_data_matrix,train_labels_matrix,test_data,task)
        all_errors[i,2] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,2] = util.dice_overlap(test_labels, predicted_labels)
        predicted_mask_3 = predicted_labels.reshape(im_size[0],im_size[1])
        ax3 = fig.add_subplot(133)
        ax3.imshow(test_shape_1, 'gray')
        ax3.imshow(predicted_mask_3, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,2], all_dice[i,2])
        ax3.set_xlabel(text_str)
        ax3.set_title('Subject {}: My method'.format(sub))
