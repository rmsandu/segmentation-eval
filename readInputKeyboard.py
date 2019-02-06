# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 11:34:48 2017

@author: 310241758
"""


def getNaturalNumber(prompt):
    return getInteger(prompt, mustBePositive=True)


def getInteger(prompt, mustBePositive=False):
    v = None
    while v is None:
        try:
            vIn = input(prompt + ':')
            v = int(vIn)
            if mustBePositive and v < 1:
                print("You must enter a POSITIVE number")
                v = None
        except ValueError:
            print("Non integer value given, please enter a valid integer")
    return v


def getNonEmptyString(prompt):
    v = None
    while v is None or len(v) == 0:
        v = input(prompt + ':').strip()
    return v


def getChoice(prompt, choices):
    assert (type(choices) == list)
    choices = list(map(lambda L: L.lower(), choices))
    clf_name = None
    while clf_name is None:
        clf = input(prompt + ' (any of ' + ",".join(choices) + '):').lower()
        if clf in choices:
            clf_name = clf
            return clf_name
        else:
            print("Oops, you did not enter a valid value from the available choices")


def getChoiceYesNo(prompt, choices):
    assert (type(choices) == list)
    choices = list(map(lambda L: L.lower(), choices))
    clf_name = None
    while clf_name is None:
        clf = input(prompt + ' (any of ' + ",".join(choices) + '):').lower()
        if clf in choices:
            if clf == 'y':
                clf_name = True
            elif clf == 'n':
                clf_name = False
            return clf_name
        else:
            print("Oops, you did not enter a valid value from the available choices")
