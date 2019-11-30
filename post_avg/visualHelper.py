# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

def plotPredStats(feats, lb, K=10, image=None, noiseImage=None, savePath=None):
    
    # score by averaging
    scores = torch.mean(feats, dim=0)
    
    # sort and select the top K scores
    hScores, hCates = torch.sort(scores, dim=0, descending=True)
    hScores = hScores[:K].numpy()
    hCates = hCates[:K].numpy()
    
    # get individual preditions
    _, preds = torch.max(feats, dim=1)
    
    # count votes
    preds_count = {lb: 0}
    for i in range(feats.size(0)):
        if preds[i].item() in preds_count:
            preds_count[preds[i].item()] = preds_count[preds[i].item()] + 1
        else:
            preds_count[preds[i].item()] = 1
            
    candidates = sorted(preds_count.keys())
    votes = [preds_count[x] for x in candidates]
    
    # generate figure
    fig = plt.figure()
    if image is None and noiseImage is None:
        ax1, ax2, ax3 = fig.subplots(3, 1)
    else:
        axes = fig.subplots(2, 2)
        ax1 = axes[0, 0]
        ax2 = axes[1, 0]
        ax3 = axes[0, 1]
        ax4 = axes[1, 1]
    
    # chart 1, votes distribution
    inx1 = list(range(len(candidates)))
    clr1 = []
    for i in inx1:
        if candidates[i] == lb:
            clr1.append('Red')
        else:
            clr1.append('SkyBlue')
    rects1 = ax1.bar(inx1, votes, color=clr1)
    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + 0.5 * rect.get_width(), 1.01 * h, '{}'.format(h), ha='center', va='bottom')
    ax1.set_ylim(top=1.1 * ax1.get_ylim()[1])
    ax1.set_xticks(inx1)
    ax1.set_xticklabels([str(x) for x in candidates], rotation=30)
    ax1.set_ylabel('votes')
    ax1.set_title('Votes Distribution')
    
    # chart 2, top prediction scores
    inx2 = list(range(len(hCates)))
    clr2 = []
    for i in inx2:
        if hCates[i] == lb:
            clr2.append('Red')
        else:
            clr2.append('SkyBlue')
    rects2 = ax2.bar(inx2, hScores, color=clr2)
    for rect in rects2:
        h = rect.get_height()
        ax2.text(rect.get_x() + 0.5 * rect.get_width(), 1.01 * h, '{:.2f}'.format(h), ha='center', va='bottom')
    ax2.set_ylim(top=1.1 * ax2.get_ylim()[1])
    ax2.set_xticks(inx2)
    ax2.set_xticklabels([str(x) for x in hCates], rotation=30)
    ax2.set_ylabel('score')
    ax2.set_xlabel('Top Predictions')
    
    # axis 3, the noise image
    if noiseImage is not None:
        ax3.imshow(noiseImage)
        ax3.set_xlabel('Noise Image')
        ax3.set_axis_off()
    else:
        # if noise image is not given, show prediction event plot
        clr3 = []
        for i in range(preds.size(0)):
            if preds[i] == lb:
                clr3.append('Red')
            else:
                clr3.append('Green')
        ax3.eventplot(preds.unsqueeze(1).numpy(), orientation='vertical', colors=clr3)
        ax3.set_yticks(candidates)
        ax3.set_yticklabels([str(x) for x in candidates])
        ax3.set_xlabel('sample index')
        ax3.set_ylabel('class')

    # axis 4, the input image
    if image is not None:
        ax4.imshow(image)
        ax4.set_title('Input Image')
        ax4.set_axis_off()
    
    # save figure and close
    if savePath is not None:
        fig.savefig(savePath)
        
    plt.close(fig)


def plotPerturbationDistribution(perturbations, savePath=None):

    # generate figure
    fig = plt.figure()
    ax1, ax2, ax3 = fig.subplots(3, 1)
    
    # plot scatter chart
    perts = np.asarray(perturbations)
    ax1.scatter(perts[:, 0], perts[:, 1], c='SkyBlue')
    ax1.autoscale(axis='x')
    ax1.set_ylim((-1, 2))
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['missed', 'defensed'])
    ax1.set_xlabel('Perturbation distance')
    ax1.set_title('Perturbations Distribution')
    
    # plot bin chart for defensed adversarial samples
    x = [e[0] for e in perturbations if e[1] == 1]
    ax2.hist(x, bins=20, color='SkyBlue')
    ax2.set_xlabel('Perturbation distance')
    ax2.set_ylabel('Denfensed')
    
    # plot bin chart for missed adversarial samples
    x = [e[0] for e in perturbations if e[1] == 0]
    ax3.hist(x, bins=20, color='Red')
    ax3.set_xlabel('Perturbation distance')
    ax3.set_ylabel('Missed')
    
    # save figure and close
    if savePath is not None:
        fig.savefig(savePath)
        
    plt.close(fig)

