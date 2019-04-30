import psmatching.match as psm
import pytest


file = "simMATCH.csv"
model = "CASE ~ AGE + TOTAL_YRS"
k = "3"

m = psm.PSMatch(file, model, k)


def test_class():
    assert m.file
    assert m.model
    assert m.k


def test_prep_data():
    global m
    m.prepare_data()
    assert not m.df.empty


def test_match():
    global m
    m.match()
    assert not m.matches.empty
    assert not m.matched_data.empty


def test_eval():
    global m
    m.prepare_data()
    m.match()
    assert m.evaluate()
