import os

import matplotlib.pyplot as plt
import numpy as np

# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


# import simulation related support
from Basilisk.simulation import spacecraft
from Basilisk.simulation import magneticFieldWMM

# general support file with common unit test functions
# import general simulation support files
from Basilisk.utilities import simHelpers
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simIncludeGravBody,
)
from Basilisk.utilities.supportDataTools.dataFetcher import get_path, DataFile

# attempt to import vizard
from Basilisk.utilities import vizSupport


def run(show_plots, orbitCase):
    """
    At the end of the python script you can specify the following example parameters.

    Args:
        show_plots (bool): Determines if the script should display plots
        orbitCase (str): {'circular', 'elliptical'}

    """

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    #
    #  create the simulation process
    #
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # create the dynamics task and specify the integration update time
    simulationTimeStep = macros.sec2nano(60.0)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    #
    #   setup the simulation tasks/objects
    #

    # initialize spacecraft object and set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"

    # add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)

    # setup Gravity Body
    gravFactory = simIncludeGravBody.gravBodyFactory()
    planet = gravFactory.createEarth()
    planet.isCentralBody = True  # ensure this is the central gravitational body
    mu = planet.mu
    req = planet.radEquator

    # attach gravity model to spacecraft
    gravFactory.addBodiesTo(scObject)

    # create the magnetic field
    magModule = magneticFieldWMM.MagneticFieldWMM()
    wmm_path = get_path(DataFile.MagneticFieldData.WMM)
    magModule.configureWMMFile(str(wmm_path))
    magModule.ModelTag = "WMM"

    # set the minReach and maxReach values if on an elliptic orbit
    if orbitCase == "elliptical":
        magModule.envMinReach = 10000 * 1000.0
        magModule.envMaxReach = 20000 * 1000.0

    # set epoch date/time message
    epochMsg = simHelpers.timeStringToGregorianUTCMsg(
        "2020 MAY 21 18:28:03 (UTC)"
    )

    # add spacecraft to the magnetic field module so it can read the sc position messages
    magModule.addSpacecraftToModel(
        scObject.scStateOutMsg
    )  # this command can be repeated if multiple

    # add the magnetic field module to the simulation task stack
    scSim.AddModelToTask(simTaskName, magModule)

    #
    #   setup orbit and simulation time
    #
    # setup the orbit using classical orbit elements
    oe = orbitalMotion.ClassicElements()
    rPeriapses = req + 500*1000
    if orbitCase == "circular":
        oe.a = rPeriapses
        oe.e = 0.0000
    elif orbitCase == "elliptical":
        rApoapses = req * 3.5
        oe.a = (rPeriapses + rApoapses) / 2.0
        oe.e = 1.0 - rPeriapses / oe.a
    else:
        print("Unsupported orbit type " + orbitCase + " selected")
        exit(1)
    oe.i = 85.0 * macros.D2R
    oe.Omega = 48.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 85.3 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    # next lines stores consistent initial orbit elements
    # with circular or equatorial orbit, some angles are arbitrary
    oe = orbitalMotion.rv2elem(mu, rN, vN)

    #
    #   initialize Spacecraft States with the initialization variables
    #
    scObject.hub.r_CN_NInit = rN  # m   - r_BN_N
    scObject.hub.v_CN_NInit = vN  # m/s - v_BN_N

    # set the simulation time
    n = np.sqrt(mu / oe.a / oe.a / oe.a)
    P = 2.0 * np.pi / n
    simulationTime = macros.sec2nano(1.0 * P)

    # connect messages
    magModule.epochInMsg.subscribeTo(epochMsg)

    #
    #   Setup data logging before the simulation is initialized
    #
    numDataPoints = 100
    samplingTime = simHelpers.samplingTime(
        simulationTime, simulationTimeStep, numDataPoints
    )
    dataLog = scObject.scStateOutMsg.recorder(samplingTime)
    magLog = magModule.envOutMsgs[0].recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, dataLog)
    scSim.AddModelToTask(simTaskName, magLog)

    # if this scenario is to interface with the BSK Viz, uncomment the following line
    if vizSupport.vizFound:
        viz = vizSupport.enableUnityVisualization(
            scSim,
            simTaskName,
            scObject,
            # saveFile=fileName,
        )
        viz.epochInMsg.subscribeTo(epochMsg)

        viz.settings.show24hrClock = 1
        viz.settings.showDataRateDisplay = 1

    scSim.InitializeSimulation()

    #
    #   configure a simulation stop time and execute the simulation run
    #
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    #
    #   retrieve the logged data
    #
    magData = magLog.magField_N
    posData = dataLog.r_BN_N

    np.set_printoptions(precision=16)

    #
    #   plot the results
    #
    # draw the inertial position vector components
    plt.close("all")  # clears out plots from earlier test runs

    timeAxis = dataLog.times() * macros.NANO2SEC
    plt.figure(1)
    fig = plt.gcf()
    ax = fig.gca()
    ax.ticklabel_format(useOffset=False, style="sci")
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    )
    rData = []
    for idx in range(0, len(posData)):
        rMag = np.linalg.norm(posData[idx])
        rData.append(rMag / 1000.0)
    plt.plot(timeAxis / P, rData, color="#aa0000")
    if orbitCase == "elliptical":
        plt.plot(
            timeAxis / P,
            [magModule.envMinReach / 1000.0] * len(rData),
            color="#007700",
            dashes=[5, 5, 5, 5],
        )
        plt.plot(
            timeAxis / P,
            [magModule.envMaxReach / 1000.0] * len(rData),
            color="#007700",
            dashes=[5, 5, 5, 5],
        )

    plt.xlabel("Time [orbits]")
    plt.ylabel("Radius [km]")
    plt.ylim(min(rData) * 0.9, max(rData) * 1.1)
    figureList = {}
    pltName = fileName + "1" + orbitCase
    figureList[pltName] = plt.figure(1)

    plt.figure(2)
    fig = plt.gcf()
    ax = fig.gca()
    ax.ticklabel_format(useOffset=False, style="sci")
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    )
    for idx in range(3):
        plt.plot(
            timeAxis / P,
            magData[:, idx] * 1e9,
            color=simHelpers.getLineColor(idx, 3),
            label=r"$B\_N_{" + str(idx) + "}$",
        )
    plt.legend(loc="lower right")
    plt.xlabel("Time [orbits]")
    plt.ylabel("Magnetic Field [nT]")
    pltName = fileName + "2" + orbitCase
    figureList[pltName] = plt.figure(2)

    if show_plots:
        plt.show()

    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    return figureList


#
# This statement below ensures that the unit test scrip can be run as a
# stand-along python script
#
if __name__ == "__main__":
    run(
        True,  # show_plots
        "circular",  # orbit Case (circular, elliptical)
    )
