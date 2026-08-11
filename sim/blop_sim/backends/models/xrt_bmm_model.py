# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2026-08-07"

Created with xrtQook


NSLS-II BMM (6-BM) beamline model, improved from bmm_final.xml.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials.elemental as rmatsel
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.figure_error as rfe
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

Si111 = rmats.crystals_basic.CrystalSi(
    a=5.4307717932001225,
    d=3.1354575567115175,
    V=160.17128543981727,
    elements=['Si'],
    quantities=[1.0],
    name=r"Si111")

Si311 = rmats.crystals_basic.CrystalSi(
    a=5.4307717932001225,
    hkl=[3, 1, 1],
    d=1.6374393054627614,
    V=160.17128543981727,
    elements=['Si'],
    quantities=[1.0],
    name=r"Si311")

pt01 = rmatsel.Pt(
    name=r"pt01",
    elements=['Pt'],
    quantities=[1.0])

rh01 = rmatsel.Rh(
    name=r"rh01",
    elements=['Rh'],
    quantities=[1.0])

si01 = rmatsel.Si(
    name=r"si01",
    elements=['Si'],
    quantities=[1.0])

rhpt01 = rmats.multilayer.Coated(
    coating=rh01,
    cThickness=50.0,
    surfaceRoughness=3.0,
    substrate=pt01,
    name=r"rhpt01")

FE_M1 = rfe.RandomRoughness(
    name=r"FE_M1",
    limPhysX=[-15.0, 15.0],
    limPhysY=[-550.0, 550.0],
    gridStep=1.0,
    rms=4.5,
    corrLength=100.0,
    seed=11,
    rmsKind=r"slope")

FE_M2 = rfe.RandomRoughness(
    name=r"FE_M2",
    limPhysX=[-15.0, 15.0],
    limPhysY=[-550.0, 550.0],
    gridStep=1.0,
    rms=4.5,
    corrLength=100.0,
    seed=22,
    rmsKind=r"slope")


def build_beamline():
    bl = raycing.BeamLine(
        name=r"BMM",
        description=None)

    bl.TPW = rsources.synchr.Wiggler(
        bl=bl,
        name=r"TPW",
        center=[0.0, 0.0, 0.0],
        eE=3.0,
        eI=0.5,
        eEspread=0.0009,
        eSigmaX=94.86832980505137,
        eSigmaZ=4.47213595499958,
        betaZ=2.0000000000000004,
        xPrimeMax=0.4,
        zPrimeMax=0.07,
        eMin=10602.0,
        eMax=10622.0,
        K=12.81,
        period=120,
        n=1,
        nrays=200000)

    bl.FE_MASK = rapts.RectangularAperture(
        bl=bl,
        name=r"FE_MASK",
        center=[0.0, 12385.0, 0.0],
        blades={'left': -40.2, 'right': 30.8, 'bottom': -10.55, 'top': 10.55},
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    bl.M1_VCM = roes.parametric.ParabolicalMirrorParam(
        p=13000,
        bl=bl,
        name=r"M1_VCM",
        center=[0.0, 13000.0, 0.0],
        pitch=0.0035,
        material=rhpt01,
        figureError=FE_M1,
        limPhysX=[-150.0, 150.0],
        limPhysY=[-550.0, 550.0],
        isParametric=True,
        order=1)

    bl.Diag1 = rscreens.Screen(
        bl=bl,
        name=r"Diag1",
        center=[0.0, 25077.0, 84.4915],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[-22.0, 22.0],
        limPhysY=[-3.0, 3.0],
        cLimits=[7102.0, 7122.0])

    bl.DCM = roes.dcm.DCM(
        bragg=0.288747096812489,
        cryst2perpTransl=15.647796985881673,
        cryst2roll=0.0004,
        # cryst2roll=0.0
        limPhysX2=[-50.0, 50.0],
        limPhysY2=[0.0, 100.0],
        material2=Si111,
        bl=bl,
        name=r"DCM",
        center=[0.0, 26105.0, 91.70508948725096],
        material=Si111,
        limPhysX=[-50.0, 50.0],
        limPhysY=[-50.0, 50.0],
        order=1)

    bl.PinkBeamStop = rapts.RectangularAperture(
        bl=bl,
        name=r"PinkBeamStop",
        center=[0.0, 26450.0, 124.07154184030665],
        blades={'left': -25, 'right': 25, 'bottom': -10, 'top': 10},
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    bl.DM2_Slits = rapts.RectangularAperture(
        bl=bl,
        name=r"DM2_Slits",
        center=[0.0, 26950.0, 127.6515],
        blades={'left': -90, 'right': 90, 'bottom': -50.65, 'top': 50.65},
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    bl.Diag2 = rscreens.Screen(
        bl=bl,
        name=r"Diag2",
        center=[0.0, 27050.0, 128.3506],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[-12.0, 12.0],
        limPhysY=[-2.0, 2.0],
        cLimits=[7102.0, 7122.0])

    bl.M2_TFM = roes.ToroidMirror(
        bl=bl,
        name=r"M2_TFM",
        center=[0.0, 28473.0, 138.25440282538193],
        pitch=-0.0034755,
        positionRoll=3.141592653589793,
        material=rhpt01,
        limPhysX=[-150.0, 150.0],
        limPhysY=[-550.0, 550.0],
        order=1,
        R=6758285.7,
        r=82.789,
        figureError=FE_M2)

    bl.M3_HRM = roes.base.OE(
        bl=bl,
        name=r"M3_HRM",
        center=[0.0, 30381.0, 137.8147686567006],
        pitch=0.0035,
        positionRoll=3.141592653589793,
        material=si01,
        limPhysX=[-150.0, 150.0],
        limPhysY=[-550.0, 550.0],
        order=1)

    bl.NANO_BPM = rscreens.Screen(
        bl=bl,
        name=r"NANO_BPM",
        center=[0.0, 31122.0, 132.2156],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[-12.0, 12.0],
        limPhysY=[-2.0, 2.0],
        cLimits=[7102.0, 7122.0])

    bl.BeamShutter = rapts.RectangularAperture(
        bl=bl,
        name=r"BeamShutter",
        center=[0.0, 31555.0, 129.1703],
        blades={'left': -100, 'right': 100, 'bottom': -130, 'top': 130},
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    bl.DM3_Slits = rapts.RectangularAperture(
        bl=bl,
        name=r"DM3_Slits",
        center=[0.0, 39500.0, 73.2937],
        blades={'left': -140.0, 'right': 140.0, 'bottom': -140.5, 'top': 140.5},
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    bl.XAS_SAMPLE = rscreens.Screen(
        bl=bl,
        name=r"XAS_SAMPLE",
        center=[0.0, 40300.0, 67.6674],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[-1000.0, 1000.0],
        limPhysY=[-700.4725, 700.4725],
        histShape=[1456.0, 1088.0])

    bl.XRD_SAMPLE = rscreens.Screen(
        bl=bl,
        name=r"XRD_SAMPLE",
        center=[0.0, 44509.0, 38.0658],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[-6.0, 6.0],
        limPhysY=[-2.0, 2.0],
        cLimits=[7102.0, 7122.0])

    return bl


def run_process(bl):
    TPW_global = bl.TPW.shine()

    FE_MASK_local = bl.FE_MASK.propagate(
        beam=TPW_global)

    M1_VCM_global, M1_VCM_local = bl.M1_VCM.reflect(
        beam=TPW_global)

    Diag1_local = bl.Diag1.expose(
        beam=M1_VCM_global,
        withHistogram=True)

    DCM_global, DCM_local1, DCM_local2 = bl.DCM.double_reflect(
        beam=M1_VCM_global)

    PinkBeamStop_local = bl.PinkBeamStop.propagate(
        beam=DCM_global)

    DM2_Slits_local = bl.DM2_Slits.propagate(
        beam=DCM_global)

    Diag2_local = bl.Diag2.expose(
        beam=DCM_global,
        withHistogram=True)

    M2_TFM_global, M2_TFM_local = bl.M2_TFM.reflect(
        beam=DCM_global)

    M3_HRM_global, M3_HRM_local = bl.M3_HRM.reflect(
        beam=M2_TFM_global)

    NANO_BPM_local = bl.NANO_BPM.expose(
        beam=M3_HRM_global,
        withHistogram=True)

    BeamShutter_local = bl.BeamShutter.propagate(
        beam=M3_HRM_global)

    DM3_Slits_local = bl.DM3_Slits.propagate(
        beam=M3_HRM_global)

    XAS_SAMPLE_local = bl.XAS_SAMPLE.expose(
        beam=M3_HRM_global,
        withHistogram=True)

    XRD_SAMPLE_local = bl.XRD_SAMPLE.expose(
        beam=M3_HRM_global,
        withHistogram=True)

    outDict = {
        'TPW_global': TPW_global,
        'FE_MASK_local': FE_MASK_local,
        'M1_VCM_global': M1_VCM_global,
        'M1_VCM_local': M1_VCM_local,
        'Diag1_local': Diag1_local,
        'DCM_global': DCM_global,
        'DCM_local1': DCM_local1,
        'DCM_local2': DCM_local2,
        'PinkBeamStop_local': PinkBeamStop_local,
        'DM2_Slits_local': DM2_Slits_local,
        'Diag2_local': Diag2_local,
        'M2_TFM_global': M2_TFM_global,
        'M2_TFM_local': M2_TFM_local,
        'M3_HRM_global': M3_HRM_global,
        'M3_HRM_local': M3_HRM_local,
        'NANO_BPM_local': NANO_BPM_local,
        'BeamShutter_local': BeamShutter_local,
        'DM3_Slits_local': DM3_Slits_local,
        'XAS_SAMPLE_local': XAS_SAMPLE_local,
        'XRD_SAMPLE_local': XRD_SAMPLE_local}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"Diag1_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-25, 25]),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-5, 5]),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"01 - Diag 1")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"Diag2_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-10, 10]),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-10, 10]),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"02 - Diag2")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"XAS_SAMPLE_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-10, 10],
            bins=728,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-7.4725, 7.4725],
            bins=544,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=544,
            ppb=1),
        title=r"03 - XAS Sample screen")
    plots.append(plot03)
    return plots

def build_histRGB(lb, gb, limits=None, isScreen=False, shape=None):
    if shape is None:
        shape = [256, 256]
    good = (lb.state == 1) | (lb.state == 2)
    if isScreen:
        x, y, z = lb.x[good], lb.z[good], lb.y[good]
    else:
        x, y, z = lb.x[good], lb.y[good], lb.z[good]
    goodlen = len(lb.x[good])
    hist2dRGB = np.zeros((shape[1], shape[0], 3), dtype=np.float64)
    hist2d = np.zeros((shape[1], shape[0]), dtype=np.float64)

    if limits is None and goodlen > 0:
        limits = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)]])

    if goodlen > 0:
        beamLimits = [limits[1], limits[0]] or None
        flux = gb.Jss[good] + gb.Jpp[good]
        hist2d, _, _ = np.histogram2d(y, x, bins=[shape[1], shape[0]], range=beamLimits, weights=flux)
        hist2dRGB = None
    return hist2d, hist2dRGB, limits


def tune_dcm(bl, E0, fixedExit=30.0, beamInclination=0.00700):
    """Tune DCM for new energy"""
    crystal = bl.DCM.material
    theta = crystal.get_Bragg_angle(E0)
    try:
        dtheta = float(np.mean(crystal.get_dtheta(E0)))
    except Exception:
        dtheta = 0.0
    bl.DCM.bragg = theta - dtheta + beamInclination
    if fixedExit is not None:
        bl.DCM.cryst2perpTransl = fixedExit / (2.0 * np.cos(bl.DCM.bragg))
    return bl.DCM.bragg


def set_energy(bl, E0, band=20.0, fixedExit=None, cryst2roll=None):
    """Change energy of beamline"""
    bl.TPW.eMin = E0 - 0.5 * band
    bl.TPW.eMax = E0 + 0.5 * band
    bl.alignE = E0
    if cryst2roll is not None:
        bl.DCM.cryst2roll = cryst2roll
    tune_dcm(bl, E0, fixedExit=fixedExit)


def main():
    ...

if __name__ == '__main__':
    main()
