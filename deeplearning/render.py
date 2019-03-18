#!/usr/bin/env python

import altair as alt
import pandas as pd

def render(f, out):
  dat = pd.read_csv(f, delimiter="\t", header=None, names=('grp', 'iter', 'dice'))
  sdat = dat[dat.grp == "train"]
  sdat2 = dat[dat.grp == "valid"]
  sdat["diceavg"] = sdat.dice.rolling(100).mean()

  chart1 = alt.Chart(sdat).mark_line().encode(
    alt.X('iter:Q', title="Iteration"),
    alt.Y('diceavg:Q', title="Dice", axis=alt.Axis(format='%'))
  )
  chart2 = alt.Chart(sdat2).mark_line(color="#ffd100", strokeWidth=5).encode(
    alt.X('iter:Q', title="Iteration"),
    alt.Y('dice:Q', title="Dice", axis=alt.Axis(format='%'))
  )
  chart = chart1 + chart2

  chart.save(out)
  return

#render("history.txt")
