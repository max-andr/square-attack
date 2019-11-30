# -*- coding: utf-8 -*-

import robustml
import numpy as np

import foolbox.criteria as crt
import foolbox.attacks as attacks
import foolbox.distances as distances
import foolbox.adversarial as adversarial

class NullAttack(robustml.attack.Attack):
    def run(self, x, y, target):
        return x

class FoolboxAttackWrapper(robustml.attack.Attack):
    def __init__(self, attack):
        self._attacker = attack
    
    def run(self, x, y, target):
        # model requires image in (C, H, W), but robustml provides (H, W, C)
        # transpose x to accommodate pytorch's axis arrangement convention
        x = np.transpose(x, (2, 0, 1))
        if target is not None:
            adv_criterion = crt.TargetClass(target)
            adv_obj = adversarial.Adversarial(self._attacker._default_model, adv_criterion, x, y, distance=self._attacker._default_distance)
            adv_x = self._attacker(adv_obj)
        else:
            adv_x = self._attacker(x, y)
        
        # transpose back to data provider's convention       
        return np.transpose(adv_x, (1, 2, 0))

def fgsmAttack(victim_model):   # victim_model should be model wrapped with foolbox model
    attacker = attacks.GradientSignAttack(victim_model, crt.Misclassification())
    return FoolboxAttackWrapper(attacker)
        
def pgdAttack(victim_model):    # victim_model should be model wrapped with foolbox model
    attacker = attacks.RandomStartProjectedGradientDescentAttack(victim_model, crt.Misclassification(), distance=distances.Linfinity)
    return FoolboxAttackWrapper(attacker)
    
def dfAttack(victim_model):   # victim_model should be model wrapped with foolbox model
    attacker = attacks.DeepFoolAttack(victim_model, crt.Misclassification())
    return FoolboxAttackWrapper(attacker)
    
def cwAttack(victim_model): # victim_model should be model wrapped with foolbox model
    attacker = attacks.CarliniWagnerL2Attack(victim_model, crt.Misclassification())
    return FoolboxAttackWrapper(attacker)
    