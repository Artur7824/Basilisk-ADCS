import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (mrpFeedback, attTrackingError, inertial3D, rwMotorTorque)
from Basilisk.simulation import (reactionWheelStateEffector, simpleNav, spacecraft, boreAngCalc)
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simIncludeGravBody,
    simIncludeRW,
    vizSupport,
)
from Basilisk.utilities import simHelpers

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])



def run(show_plots, use2SunSensors, starTrackerFov, sunSensorFov, attitudeSetCase):

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    scSim = SimulationBaseClass.SimBaseClass()


    scSim.SetProgressBar(False)

    dynProcess = scSim.CreateNewProcess(simProcessName)

    simulationTime = macros.min2nano(1.5)
    simulationTimeStep = macros.sec2nano(0.5)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "Bevo2-Sat"

    scSim.AddModelToTask(simTaskName, scObject, 1)
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravBodies = gravFactory.createBodies('earth', 'sun')
    planet = gravBodies['earth']
    planet.isCentralBody = True

    mu = planet.mu
    gravFactory.addBodiesTo(scObject)

    timeInitString = "2021 JANUARY 15 00:28:30.0"
    spiceTimeStringFormat = '%Y %B %d %H:%M:%S.%f'
    timeInit = datetime.strptime(timeInitString, spiceTimeStringFormat)

    spiceObject = gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)

    spiceObject.zeroBase = 'Earth'

    scSim.AddModelToTask(simTaskName, spiceObject, 2)

    gravFactory.addBodiesTo(scObject)

    # setup the orbit using classical orbit elements
    oe = orbitalMotion.ClassicElements()
    oe.a = 7000. * 1000      # meters
    oe.e = 0.0001
    oe.i = 33.3 * macros.D2R
    oe.Omega = 148.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 135 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    oe = orbitalMotion.rv2elem(mu, rN, vN)

    sigma_BN_start = [ [0.522, -0.065,  0.539],     # to violate one keepIn only
                       [0.314, -0.251,  0.228],     # to violate two keepIn and not keepOut
                       [-0.378, 0.119, -0.176],     # to violate keepOut and both keepIn
                       [-0.412, 0.044, -0.264] ]    # to violate keepOut only

    scObject.hub.r_CN_NInit = rN  # m   - r_BN_N
    scObject.hub.v_CN_NInit = vN  # m/s - v_BN_N
    scObject.hub.sigma_BNInit = sigma_BN_start[attitudeSetCase]   # change this MRP set to customize initial inertial attitude
    scObject.hub.omega_BN_BInit = [[0.], [0.], [0.]]              # rad/s - omega_CN_B

    I = [0.02 / 3,  0.,         0.,
         0.,        0.1256 / 3, 0.,
         0.,        0.,         0.1256 / 3]
    scObject.hub.mHub = 4.0  # kg - spacecraft mass
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(I)

    rwFactory = simIncludeRW.rwFactory()

    varRWModel = messaging.BalancedWheels

    maxMomentum = 0.01
    maxSpeed = 6000.  # [RPM]  rwFactory.create() expects Omega_max in RPM and converts internally
    RW1 = rwFactory.create('custom', [1, 0, 0], Omega=0.  # RPM
                           , Omega_max=maxSpeed
                           , maxMomentum=maxMomentum
                           , u_max=0.001
                           , RWModel=varRWModel)
    RW2 = rwFactory.create('custom', [0, 1, 0], Omega=0.  # RPM
                           , Omega_max=maxSpeed
                           , maxMomentum=maxMomentum
                           , u_max=0.001
                           , RWModel=varRWModel)
    RW3 = rwFactory.create('custom', [0, 0, 1], Omega=0.  # RPM
                           , Omega_max=maxSpeed
                           , maxMomentum=maxMomentum
                           , u_max=0.001
                           , RWModel=varRWModel)

    numRW = rwFactory.getNumOfDevices()

    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)

    scSim.AddModelToTask(simTaskName, rwStateEffector, 2)

    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject)

  	# sets of initial attitudes that yield the desired constraint violations (attitudeSetCase)
    sigma_BN_target = [ [0.342,  0.223, -0.432],     # to violate one keepIn only
                        [0.326, -0.206, -0.823],     # to violate two keepIn and not keepOut
                        [0.350,  0.220, -0.440],     # to violate keepOut and both keepIn
                        [0.350,  0.220, -0.440] ]    # to violate keepOut only

    # setup inertial3D guidance module
    inertial3DObj = inertial3D.inertial3D()
    inertial3DObj.ModelTag = "inertial3D"
    scSim.AddModelToTask(simTaskName, inertial3DObj)
    inertial3DObj.sigma_R0N = sigma_BN_target[attitudeSetCase]     # change this MRP set to customize final inertial attitude

    # setup the attitude tracking error evaluation module
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attError)

    # setup the MRP Feedback control module
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    scSim.AddModelToTask(simTaskName, mrpControl)
    decayTime = 10.0
    xi = 1.0
    mrpControl.Ki = -1  # make value negative to turn off integral feedback
    mrpControl.P = 3*np.max(I)/decayTime
    mrpControl.K = (mrpControl.P/xi)*(mrpControl.P/xi)/np.max(I)
    mrpControl.integralLimit = 2. / mrpControl.Ki * 0.1

    # add module that maps the Lr control torque into the RW motor torques
    rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
    rwMotorTorqueObj.ModelTag = "rwMotorTorque"
    scSim.AddModelToTask(simTaskName, rwMotorTorqueObj)

    # Make the RW control all three body axes
    controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    rwMotorTorqueObj.controlAxes_B = controlAxes_B

    # Boresight vector modules.
    stBACObject = boreAngCalc.BoreAngCalc()
    stBACObject.ModelTag = "starTrackerBoresight"
    stBACObject.boreVec_B = [1., 0., 0.]  # boresight in body frame
    scSim.AddModelToTask(simTaskName, stBACObject)

    ssyBACObject = boreAngCalc.BoreAngCalc()
    ssyBACObject.ModelTag = "SunSensorBoresight"
    ssyBACObject.boreVec_B = [0., 1., 0.]  # boresight in body frame
    scSim.AddModelToTask(simTaskName, ssyBACObject)

    if use2SunSensors:
        sszBACObject = boreAngCalc.BoreAngCalc()
        sszBACObject.ModelTag = "SunSensorBoresight"
        sszBACObject.boreVec_B = [0., 0., 1.]  # boresight in body frame
        scSim.AddModelToTask(simTaskName, sszBACObject)

    #
    #   Setup data logging before the simulation is initialized
    #
    numDataPoints = 200
    samplingTime = simHelpers.samplingTime(simulationTime, simulationTimeStep, numDataPoints)
    dataRec = scObject.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, dataRec)
    rwMotorLog = rwMotorTorqueObj.rwMotorTorqueOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, rwMotorLog)
    attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, attErrorLog)
    stBACOLog = stBACObject.angOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, stBACOLog)
    ssyBACOLog = ssyBACObject.angOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, ssyBACOLog)
    if use2SunSensors:
        sszBACOLog = sszBACObject.angOutMsg.recorder(samplingTime)
        scSim.AddModelToTask(simTaskName, sszBACOLog)

    # To log the RW information, the following code is used:
    mrpLog = rwStateEffector.rwSpeedOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, mrpLog)

    # A message is created that stores an array of the Omega wheel speeds
    rwLogs = []
    for item in range(numRW):
        rwLogs.append(rwStateEffector.rwOutMsgs[item].recorder(samplingTime))
        scSim.AddModelToTask(simTaskName, rwLogs[item])

    #
    # create simulation messages
    #

    # create the FSW vehicle configuration message
    # use the same inertia in the FSW algorithm as in the simulation
    vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B=I)
    vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    # Setup the FSW RW configuration message.
    fswRwParamMsg = rwFactory.getConfigMessage()

    # link messages
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)    # for inertial3D
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
    mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    mrpControl.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)
    rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)

    # Boresight modules
    stBACObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    stBACObject.celBodyInMsg.subscribeTo(spiceObject.planetStateOutMsgs[1])
    ssyBACObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    ssyBACObject.celBodyInMsg.subscribeTo(spiceObject.planetStateOutMsgs[1])
    if use2SunSensors:
        sszBACObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
        sszBACObject.celBodyInMsg.subscribeTo(spiceObject.planetStateOutMsgs[1])

    # Vizard Visualization Option
    # ---------------------------
    if vizSupport.vizFound:
        viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject,
                 # saveFile=__file__
                 )
        vizSupport.createConeInOut(viz, toBodyName='sun_planet_data', coneColor = 'r',
                           normalVector_B=[1, 0, 0], incidenceAngle=starTrackerFov*macros.D2R, isKeepIn=False,
                           coneHeight=10.0, coneName='sunKeepOut')
        vizSupport.createConeInOut(viz, toBodyName='sun_planet_data', coneColor = 'g',
                           normalVector_B=[0, 1, 0], incidenceAngle=sunSensorFov*macros.D2R, isKeepIn=True,
                           coneHeight=10.0, coneName='sunKeepIn')
        if use2SunSensors:
            vizSupport.createConeInOut(viz, toBodyName='sun_planet_data', coneColor = 'b',
                           normalVector_B=[0, 0, 1], incidenceAngle=sunSensorFov*macros.D2R, isKeepIn=True,
                           coneHeight=10.0, coneName='sunKeepIn')

    # initialize Simulation:  This function runs the self_init()
    # cross_init() and reset() routines on each module.
    scSim.InitializeSimulation()

    # configure a simulation stop time and execute the simulation run
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    # retrieve the logged data

    # Orbit
    posData = dataRec.r_BN_N
    velData = dataRec.v_BN_N

    # RW
    dataUsReq = rwMotorLog.motorTorque
    dataSigmaBR = attErrorLog.sigma_BR
    dataOmegaBR = attErrorLog.omega_BR_B
    dataOmegaRW = mrpLog.wheelSpeeds
    dataSTMissAngle = stBACOLog.missAngle
    dataSSyMissAngle = ssyBACOLog.missAngle
    if use2SunSensors:
        dataSSzMissAngle = sszBACOLog.missAngle

    dataRW = []
    for i in range(numRW):
        dataRW.append(rwLogs[i].u_current)

    np.set_printoptions(precision=16)


    # Displays the plots relative to the S/C attitude, maneuver, RW speeds and torques and boresight angles

    timeData = rwMotorLog.times() * macros.NANO2MIN

    plot_attitude_error(timeData, dataSigmaBR)
    figureList = {}
    pltName = fileName + "1" + str(int(use2SunSensors)) + str(starTrackerFov) + str(sunSensorFov) + str(attitudeSetCase)
    figureList[pltName] = plt.figure(1)

    plot_rw_motor_torque(timeData, dataUsReq, dataRW, numRW)
    pltName = fileName + "2" + str(int(use2SunSensors)) + str(starTrackerFov) + str(sunSensorFov) + str(attitudeSetCase)
    figureList[pltName] = plt.figure(2)

    plot_rate_error(timeData, dataOmegaBR)
    pltName = fileName + "3" + str(int(use2SunSensors)) + str(starTrackerFov) + str(sunSensorFov) + str(attitudeSetCase)
    figureList[pltName] = plt.figure(3)

    plot_rw_speeds(timeData, dataOmegaRW, numRW)
    pltName = fileName + "4" + str(int(use2SunSensors)) + str(starTrackerFov) + str(sunSensorFov) + str(attitudeSetCase)
    figureList[pltName] = plt.figure(4)

    plot_st_miss_angle(timeData, dataSTMissAngle, starTrackerFov)
    pltName = fileName + "5" + str(int(use2SunSensors)) + str(starTrackerFov) + str(sunSensorFov) + str(attitudeSetCase)
    figureList[pltName] = plt.figure(5)

    dataSS = [dataSSyMissAngle]
    if use2SunSensors:
        dataSS.append(dataSSzMissAngle)
    plot_ss_miss_angle(timeData, dataSS, sunSensorFov)
    pltName = fileName + "6" + str(int(use2SunSensors)) + str(starTrackerFov) + str(sunSensorFov) + str(attitudeSetCase)
    figureList[pltName] = plt.figure(6)

    if show_plots:
        plt.show()

    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    return figureList


# Plotting RW functions
def plot_attitude_error(timeData, dataSigmaBR):
    """Plot the attitude errors."""
    plt.figure(1)
    for idx in range(3):
        plt.plot(timeData, dataSigmaBR[:, idx],
                 color=simHelpers.getLineColor(idx, 3),
                 label=r'$\sigma_' + str(idx) + '$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Attitude Error $\sigma_{B/R}$')

def plot_rw_cmd_torque(timeData, dataUsReq, numRW):
    """Plot the RW command torques."""
    plt.figure(2)
    for idx in range(3):
        plt.plot(timeData, dataUsReq[:, idx],
                 '--',
                 color=simHelpers.getLineColor(idx, numRW),
                 label=r'$\hat u_{s,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('RW Motor Torque (Nm)')

def plot_rw_motor_torque(timeData, dataUsReq, dataRW, numRW):
    """Plot the RW actual motor torques."""
    plt.figure(2)
    for idx in range(3):
        plt.plot(timeData, dataUsReq[:, idx],
                 '--',
                 color=simHelpers.getLineColor(idx, numRW),
                 label=r'$\hat u_{s,' + str(idx) + '}$')
        plt.plot(timeData, dataRW[idx],
                 color=simHelpers.getLineColor(idx, numRW),
                 label='$u_{s,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('RW Motor Torque (Nm)')

def plot_rate_error(timeData, dataOmegaBR):
    """Plot the body angular velocity rate tracking errors."""
    plt.figure(3)
    for idx in range(3):
        plt.plot(timeData, dataOmegaBR[:, idx],
                 color=simHelpers.getLineColor(idx, 3),
                 label=r'$\omega_{BR,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Rate Tracking Error (rad/s) ')

def plot_rw_speeds(timeData, dataOmegaRW, numRW):
    """Plot the RW spin rates."""
    plt.figure(4)
    for idx in range(numRW):
        plt.plot(timeData, dataOmegaRW[:, idx] / macros.RPM,
                 color=simHelpers.getLineColor(idx, numRW),
                 label=r'$\Omega_{' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('RW Speed (RPM) ')

def plot_st_miss_angle(timeData, dataMissAngle, Fov):
    """Plot the miss angle between star tacker boresight and Sun."""
    fig, ax = plt.subplots()
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    dataFov = np.ones(len(timeData))*Fov
    plt.plot(timeData, dataFov, '--',
                 color = 'r', label = r'f.o.v.')
    data = dataMissAngle*macros.R2D
    for idx in range(1):
        plt.plot(timeData, data,
                 color=simHelpers.getLineColor(idx, 3),
                 label=r'$\alpha $')
    plt.fill_between(timeData, 0, 1, where=dataFov >= data, facecolor='red',
                  alpha = 0.4, interpolate=True, transform = trans)
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('StarTracker/Sun angle \u03B1 (deg)')

def plot_ss_miss_angle(timeData, dataMissAngle, Fov):
    """Plot the miss angle between sun sensor(s) boresight and Sun."""
    fig, ax = plt.subplots()
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    dataFov = np.ones(len(timeData))*Fov
    plt.plot(timeData, dataFov, '--',
                 color = 'r', label = r'f.o.v.')
    data = []
    for d in dataMissAngle:
        data.append(d*macros.R2D)
    for idx in range(len(data)):
        plt.plot(timeData, data[idx],
                 color=simHelpers.getLineColor(idx, 3),
                 label=r'$\beta_{' + str(idx+1) + '}$')
    dataMinAngle = 180*np.ones(len(timeData))
    for i in range(len(timeData)):
        for j in range(len(data)):
            if data[j][i] < dataMinAngle[i]:
                dataMinAngle[i] = data[j][i]
    plt.fill_between(timeData, 0, 1, where = dataFov < dataMinAngle, facecolor='red',
                  alpha = 0.4, interpolate=True, transform = trans)
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('SunSensor/Sun angle \u03B2 (deg)')



if __name__ == "__main__":
    run(
        True,           # show_plots
        True,           # use2SunSensors
        20,             # starTrackerFov
        70,             # sunSensorFov
        2               # attitudeSetCase
    )
