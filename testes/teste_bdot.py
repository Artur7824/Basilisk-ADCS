#Código de teste do controle B-Dot
#derivado do teste5

import os

import matplotlib.pyplot as plt
import numpy as np

from Basilisk.architecture import messaging, sysModel

from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    RigidBodyKinematics as rbk,
    orbitalMotion,
    simHelpers,
    simIncludeGravBody,
    vizSupport,
)
from Basilisk.utilities.supportDataTools.dataFetcher import get_path, DataFile

from Basilisk.simulation import (
    GravityGradientEffector,
    msisAtmosphere,
    simpleNav,
    spacecraft,
    zeroWindModel,
    facetDragDynamicEffector,
    facetSRPDynamicEffector,
    eclipse,
    meanRevertingNoiseStateEffector,
    svIntegrators,
    magneticFieldWMM,
    magnetometer,
    MtbEffector,
    starTracker,
    boreAngCalc,
    imuSensor,
)

from Basilisk.fswAlgorithms import (
    attTrackingError,
    hillPoint,
    tamComm,
)

#m_cmd = -K * dB/dt
class BdotControl(sysModel.SysModel):
    """
    Entrada : tamInMsg        -> TAMSensorBodyMsg (saída do tamComm, campo tam_B [T])
    Saída   : dipoleCmdOutMsg -> MTBCmdMsg (campo mtbDipoleCmds [A*m^2]), lida pelo MtbEffector

    Adaptações em relação à versão anterior:
      - O leitor/mensagem de entrada agora é TAMSensorBodyMsg
      - O vetor mtbDipoleCmds tem tamanho fixo (MAX_EFF_CNT); preenchido por completo.
      - Filtro passa-baixa de 1ª ordem em dB/dt: sem ele, o ruído do TAM (0.6 uT)
        diferenciado a 10 Hz domina o sinal quando a rotação é baixa.
    """

    def __init__(self, kGain, maxDipole, filterTau=0.5):
        super().__init__()
        self.ModelTag = "BdotControl"
        self.kGain = kGain              # [A*m^2 / (T/s)]
        self.maxDipole = maxDipole      # [A*m^2] por eixo
        self.filterTau = filterTau      # [s] constante de tempo do filtro de dB/dt

        self.tamInMsg = messaging.TAMSensorBodyMsgReader()
        self.dipoleCmdOutMsg = messaging.MTBCmdMsg()

        self.lastMagField = None
        self.lastTimeNanos = None
        self.bDotFiltered = np.zeros(3)

    def Reset(self, CurrentSimNanos):
        self.lastMagField = None
        self.lastTimeNanos = None
        self.bDotFiltered = np.zeros(3)
        self.dipoleCmdOutMsg.write(messaging.MTBCmdMsgPayload(), CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        currentMagField = np.array(self.tamInMsg().tam_B)  # [T], referencial do corpo
        cmdPayload = messaging.MTBCmdMsgPayload()

        if self.lastMagField is not None:
            dt = (CurrentSimNanos - self.lastTimeNanos) * macros.NANO2SEC
            if dt > 0.0:
                bDot = (currentMagField - self.lastMagField) / dt

                # Filtro passa-baixa de 1ª ordem
                alpha = dt / (self.filterTau + dt)
                self.bDotFiltered += alpha * (bDot - self.bDotFiltered)

                # Lei B-dot
                dipole = -self.kGain * self.bDotFiltered
                dipole = np.clip(dipole, -self.maxDipole, self.maxDipole)

                cmd = np.zeros(len(cmdPayload.mtbDipoleCmds))
                cmd[:3] = dipole
                cmdPayload.mtbDipoleCmds = cmd.tolist()

        self.lastMagField = currentMagField
        self.lastTimeNanos = CurrentSimNanos

        self.dipoleCmdOutMsg.write(cmdPayload, CurrentSimNanos, self.moduleID)

class StochasticAtmoBridge(sysModel.SysModel):
    def __init__(self, scObject, atmoModule, stateName):
        super().__init__()
        self.ModelTag = "StochasticAtmoBridge"
        self.scObject = scObject
        self.atmoModule = atmoModule
        self.stateName = stateName
        self.stateObject = None

        self.stochAtmoPayload = messaging.AtmoPropsMsgPayload()
        self.stochAtmoMsg = messaging.AtmoPropsMsg()

    def Reset(self, CurrentSimNanos):
        self.stateObject = None
        self.stochAtmoPayload = messaging.AtmoPropsMsgPayload()
        self.stochAtmoMsg.write(self.stochAtmoPayload, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        if self.stateObject is None:
            self.stateObject = self.scObject.dynManager.getStateObject(self.stateName)

        delta_rho = float(np.array(self.stateObject.getState()).flatten()[0])
        nominalDensity = self.atmoModule.envOutMsgs[0].read().neutralDensity
        self.stochAtmoPayload.neutralDensity = max(0.0, nominalDensity * (1.0 + delta_rho))

        self.stochAtmoMsg.write(self.stochAtmoPayload, CurrentSimNanos, self.moduleID)


class StochasticSolarBridge(sysModel.SysModel):
    def __init__(self, scObject, sunStateMsg, stateName, nominalSolarFlux=1361.0):
        super().__init__()
        self.ModelTag = "StochasticSolarBridge"
        self.scObject = scObject
        self.sunStateMsg = sunStateMsg
        self.stateName = stateName
        self.nominalSolarFlux = nominalSolarFlux
        self.stateObject = None

        self.stochSolarPayload = messaging.SolarFluxMsgPayload()
        self.stochSolarMsg = messaging.SolarFluxMsg()

    def Reset(self, CurrentSimNanos):
        self.stateObject = None
        self.stochSolarPayload.flux = self.nominalSolarFlux
        self.stochSolarMsg.write(self.stochSolarPayload, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        if self.stateObject is None:
            self.stateObject = self.scObject.dynManager.getStateObject(self.stateName)

        delta_solar = float(np.array(self.stateObject.getState()).flatten()[0])
        stoch_flux = max(0.0, self.nominalSolarFlux * (1.0 + delta_solar))
        self.stochSolarPayload.flux = stoch_flux

        self.stochSolarMsg.write(self.stochSolarPayload, CurrentSimNanos, self.moduleID)

simProcessName = "simProcess"
simTaskName = "simTask"
tamTaskName = "100hzTask"
fswTaskName = "fswTask"
stTaskName = "5hzTask"

scSim = SimulationBaseClass.SimBaseClass()
dynProcess = scSim.CreateNewProcess(simProcessName)

simulationTimeStep = macros.sec2nano(0.1) # aumentar para 1kHz
tamTimeStep = macros.sec2nano(0.1)        # aumentar para 100Hz
fswTimeStep = macros.sec2nano(0.1)        # manter em 10Hz
stTimeStep = macros.sec2nano(0.2)         # manter em 5Hz

dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))
dynProcess.addTask(scSim.CreateNewTask(tamTaskName, tamTimeStep))
dynProcess.addTask(scSim.CreateNewTask(fswTaskName, fswTimeStep))
dynProcess.addTask(scSim.CreateNewTask(stTaskName, stTimeStep))

scSim.SetProgressBar(True)

scObject = spacecraft.Spacecraft()
scObject.ModelTag = "cimatelite"

hubMass = 4.0  # [kg]
scObject.hub.mHub = hubMass

hubWidthX = 0.1   # [m]
hubWidthY = 0.1   # [m]
hubHeightZ = 0.3  # [m]

hubAreaX = hubWidthY * hubHeightZ
hubAreaY = hubWidthX * hubHeightZ
hubAreaZ = hubWidthX * hubWidthY

hubHalfX = hubWidthX / 2.0
hubHalfY = hubWidthY / 2.0
hubHalfZ = hubHeightZ / 2.0

hubI11 = hubMass / 12.0 * (hubWidthY ** 2 + hubHeightZ ** 2)
hubI22 = hubMass / 12.0 * (hubWidthX ** 2 + hubHeightZ ** 2)
hubI33 = hubMass / 12.0 * (hubWidthX ** 2 + hubWidthY ** 2)

I = [hubI11, 0.0, 0.0,
     0.0, hubI22, 0.0,
     0.0, 0.0, hubI33]

scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
scObject.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(I)

# Condições iniciais
scObject.hub.sigma_BNInit = [[0.8], [-1.2], [1.0]]  # MRP
scObject.hub.omega_BN_BInit = [
    [60.0 * macros.D2R],
    [40.0 * macros.D2R],
    [50.0 * macros.D2R],]  # [rad/s]

#definição dos corpos celestes (Terra e Sol) e interface SPICE
gravFactory = simIncludeGravBody.gravBodyFactory()
bodies = gravFactory.createBodies('earth', 'sun')

planet = bodies['earth']
planet.isCentralBody = True
sun = bodies['sun']

ggm03s_full_path = get_path(DataFile.LocalGravData.GGM03S)
planet.useSphericalHarmonicsGravityModel(str(ggm03s_full_path), 10)
mu = planet.mu

gravFactory.addBodiesTo(scObject)

rLEO = planet.radEquator + 500.0 * 1000.0
oe = orbitalMotion.ClassicElements()
oe.a = rLEO
oe.e = 0.0001
oe.i = 33.3 * macros.D2R
oe.Omega = 48.2 * macros.D2R
oe.omega = 347.8 * macros.D2R
oe.f = 85.3 * macros.D2R

rN, vN = orbitalMotion.elem2rv(mu, oe)
scObject.hub.r_CN_NInit = rN
scObject.hub.v_CN_NInit = vN

spiceObject = gravFactory.createSpiceInterface(
    time="2020 MAY 21 18:28:03 (UTC)",
    epochInMsg=True,
)
spiceObject.zeroBase = "Earth"
scSim.AddModelToTask(simTaskName, spiceObject, ModelPriority=100)

earthStateMsg = spiceObject.planetStateOutMsgs[0]  # Terra
sunStateMsg = spiceObject.planetStateOutMsgs[1]    # Sol

scSim.AddModelToTask(simTaskName, scObject)

integrator = svIntegrators.svStochasticIntegratorSOSRA(scObject)
scObject.setIntegrator(integrator)

n = np.sqrt(mu / oe.a / oe.a / oe.a)
P = 2.0 * np.pi / n

# CAMPO MAGNÉTICO WMM + MAGNETÔMETRO (MPU9250) + tamComm
magModule = magneticFieldWMM.MagneticFieldWMM()
wmm_path = get_path(DataFile.MagneticFieldData.WMM)
magModule.configureWMMFile(str(wmm_path))
magModule.ModelTag = "WMM"

magModule.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, magModule)

magModule.epochInMsg.subscribeTo(gravFactory.epochMsg)

TAM = magnetometer.Magnetometer()
TAM.ModelTag = "TAM_sensor"
TAM.scaleFactor = 1.0

TAM.senNoiseStd = [600e-9, 600e-9, 600e-9]  # 0.6 uT
TAM.senBias = [0, 0, -1e-6]
TAM.maxOutput = 4.8e-3   #  4800 uT
TAM.minOutput = -4.8e-3  # -4800 uT

tamCommObj = tamComm.tamComm()
tamCommObj.dcm_BS = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
tamCommObj.ModelTag = "tamComm"

TAM.stateInMsg.subscribeTo(scObject.scStateOutMsg)
TAM.magInMsg.subscribeTo(magModule.envOutMsgs[0])
tamCommObj.tamInMsg.subscribeTo(TAM.tamDataOutMsg)

scSim.AddModelToTask(tamTaskName, TAM)
scSim.AddModelToTask(simTaskName, tamCommObj)

#magnetorquer ~ HT-MTQ200
mtbEff = MtbEffector.MtbEffector()
mtbEff.ModelTag = "MtbEff"
scObject.addDynamicEffector(mtbEff)
scSim.AddModelToTask(simTaskName, mtbEff)

mtbConfigParams = messaging.MTBArrayConfigMsgPayload()
mtbConfigParams.numMTB = 3
maxDipole = 1.0  # [A*m^2] valor do teste5.py (ver observação no resumo)
mtbConfigParams.GtMatrix_B = [1.0, 0., 0.,
                              0., 1.0, 0.,
                              0., 0., 1.0]

mtbConfigParams.maxMtbDipoles = [maxDipole] * mtbConfigParams.numMTB
mtbParamsInMsg = messaging.MTBArrayConfigMsg().write(mtbConfigParams)

ImuSensor = imuSensor.ImuSensor()
ImuSensor.ModelTag = "imusensor"
ImuSensor.sensorPos_B = [0.0, 0.0, 0.0]

yaw = 0.0
pitch = 0.0
roll = 0.0

ImuSensor.setBodyToPlatformDCM(yaw, pitch, roll)

ImuSensor.senRotMax = 2000.0 * macros.D2R

gyro_bias = 5.0 * macros.D2R  # bias máximo
ImuSensor.senRotBias = [gyro_bias, -gyro_bias * 0.7, gyro_bias / 2]

gyro_std = 0.1 * macros.D2R
ImuSensor.PMatrixGyro = [[gyro_std, 0.0, 0.0],
                         [0.0, gyro_std, 0.0],
                         [0.0, 0.0, gyro_std]]

walk_gyro = 30.0 * macros.D2R
ImuSensor.setErrorBoundsGyro = np.array(walk_gyro)

g_accel = 9.80665
ImuSensor.senTransMax = 16.0 * g_accel

accel_bias_xy = 0.060 * g_accel
accel_bias_z = 0.080 * g_accel
ImuSensor.senTransBias = [-accel_bias_xy, accel_bias_xy / 2, accel_bias_z]

accel_std = 0.008 * g_accel
ImuSensor.PMatrixAccel = [[accel_std, 0.0, 0.0],
                          [0.0, accel_std, 0.0],
                          [0.0, 0.0, accel_std]]

walk_accel = 0.0015 * g_accel
ImuSensor.setErrorBoundsAccel = [walk_accel, walk_accel, walk_accel]

ImuSensor.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, ImuSensor)

#atmosgera MSIS + ruído
atmo = msisAtmosphere.MsisAtmosphere()
atmo.ModelTag = "MsisAtmo"

ap = 8
f107 = 110
sw_msg = {
    "ap_24_0": ap, "ap_3_0": ap, "ap_3_-3": ap, "ap_3_-6": ap, "ap_3_-9": ap,
    "ap_3_-12": ap, "ap_3_-15": ap, "ap_3_-18": ap, "ap_3_-21": ap, "ap_3_-24": ap,
    "ap_3_-27": ap, "ap_3_-30": ap, "ap_3_-33": ap, "ap_3_-36": ap, "ap_3_-39": ap,
    "ap_3_-42": ap, "ap_3_-45": ap, "ap_3_-48": ap, "ap_3_-51": ap, "ap_3_-54": ap,
    "ap_3_-57": ap, "f107_1944_0": f107, "f107_24_-24": f107,
}

swMsgList = []
for c, val in enumerate(sw_msg.values()):
    swMsgData = messaging.SwDataMsgPayload(dataValue=val)
    swMsgList.append(messaging.SwDataMsg().write(swMsgData))
    atmo.swDataInMsgs[c].subscribeTo(swMsgList[-1])

atmo.planetPosInMsg.subscribeTo(earthStateMsg)
atmo.epochInMsg.subscribeTo(gravFactory.epochMsg)
atmo.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, atmo, ModelPriority=90)

stochasticNoise = meanRevertingNoiseStateEffector.MeanRevertingNoiseStateEffector()
stochasticNoise.ModelTag = "StochasticExpAtmo"
stochasticNoise.setStationaryStd(0.15)
stochasticNoise.setTimeConstant(10 * 60.0)  # [s]
scObject.addStateEffector(stochasticNoise)

atmoBridge = StochasticAtmoBridge(scObject, atmo, stochasticNoise.getStateName())
scSim.AddModelToTask(simTaskName, atmoBridge, ModelPriority=80)

windModel = zeroWindModel.ZeroWindModel()
windModel.ModelTag = "ZeroWind"
windModel.planetPosInMsg.subscribeTo(earthStateMsg)
windModel.epochInMsg.subscribeTo(gravFactory.epochMsg)
windModel.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, windModel)

# SRP + ruído
SRPEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
SRPEffector.ModelTag = "SRPEffector"
SRPEffector.setNumFacets(6)

stochasticSolarNoise = meanRevertingNoiseStateEffector.MeanRevertingNoiseStateEffector()
stochasticSolarNoise.ModelTag = "StochasticSolarNoise"
stochasticSolarNoise.setStationaryStd(0.01)
stochasticSolarNoise.setTimeConstant(10.0 * 60.0)
scObject.addStateEffector(stochasticSolarNoise)

solarBridge = StochasticSolarBridge(
    scObject=scObject,
    sunStateMsg=sunStateMsg,
    stateName=stochasticSolarNoise.getStateName(),
    nominalSolarFlux=1361.0,
)
scSim.AddModelToTask(simTaskName, solarBridge, ModelPriority=85)

#facets
dragCoeff = 2.2
sigSpec = 0.068  # GaInP/GaAs/Ge on Ge substrate
sigDiff = 0.012

hubDrag = facetDragDynamicEffector.FacetDragDynamicEffector()
hubDrag.ModelTag = "hubDrag"


def addCubesatFacets(facetDrag, facetSRP):
    halfDims = [hubHalfX, hubHalfY, hubHalfZ]
    areas = [hubAreaX, hubAreaY, hubAreaZ]

    dcm_identity = [[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]

    for axis in (0, 1, 2):
        for sign in (+1.0, -1.0):
            normal_list = [0.0, 0.0, 0.0]
            normal_list[axis] = sign

            location_list = [0.0, 0.0, 0.0]
            location_list[axis] = sign * halfDims[axis]

            rot_list = [0.0, 0.0, 0.0]

            facetDrag.addFacet(areas[axis], dragCoeff,
                               np.array(normal_list), np.array(location_list))

            facetSRP.addFacet(
                areas[axis],    # 1. Área [m^2]
                dcm_identity,   # 2. Orientação da faceta [3x3]
                normal_list,    # 3. Normal [1x3]
                rot_list,       # 4. Eixo de rotação [1x3]
                location_list,  # 5. Localização / centro de pressão [1x3]
                sigDiff,        # 6. Coeficiente difuso
                sigSpec,        # 7. Coeficiente especular
            )


addCubesatFacets(hubDrag, SRPEffector)

#aerodrag
hubDrag.atmoDensInMsg.subscribeTo(atmoBridge.stochAtmoMsg)
hubDrag.windVelInMsg.subscribeTo(windModel.envOutMsgs[0])
scObject.addDynamicEffector(hubDrag)
scSim.AddModelToTask(simTaskName, hubDrag)

# Eclipse
eclipseObject = eclipse.Eclipse()
eclipseObject.ModelTag = "EarthEclipse"

eclipseObject.addSpacecraftToModel(scObject.scStateOutMsg)
eclipseObject.addPlanetToModel(earthStateMsg)    # corpo eclipsante: Terra (SPICE [0])
eclipseObject.sunInMsg.subscribeTo(sunStateMsg)  # fonte luminosa: Sol (SPICE [1])
scSim.AddModelToTask(simTaskName, eclipseObject)

# Conexões do SRP (idênticas ao teste5.py)
SRPEffector.sunEclipseInMsg.subscribeTo(eclipseObject.eclipseOutMsgs[0])
SRPEffector.sunInMsg.subscribeTo(sunStateMsg)
if hasattr(SRPEffector, "solarFluxInMsg"):
    SRPEffector.solarFluxInMsg.subscribeTo(solarBridge.stochSolarMsg)

scObject.addDynamicEffector(SRPEffector)
scSim.AddModelToTask(simTaskName, SRPEffector, ModelPriority=70)

# Gradiente de gravidade
ggEff = GravityGradientEffector.GravityGradientEffector()
ggEff.ModelTag = scObject.ModelTag
ggEff.addPlanetName(planet.planetName)
scObject.addDynamicEffector(ggEff)
scSim.AddModelToTask(simTaskName, ggEff)

# navigation+guidance
sNavObject = simpleNav.SimpleNav()
sNavObject.ModelTag = "SimpleNavigation"
scSim.AddModelToTask(fswTaskName, sNavObject)

attGuidance = hillPoint.hillPoint()
attGuidance.ModelTag = "hillPoint"
scSim.AddModelToTask(fswTaskName, attGuidance)

attError = attTrackingError.attTrackingError()
attError.ModelTag = "attError"
scSim.AddModelToTask(fswTaskName, attError)

attError.sigma_R0R = [0, 1, 0]

sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
attGuidance.transNavInMsg.subscribeTo(sNavObject.transOutMsg)
attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
attError.attRefInMsg.subscribeTo(attGuidance.attRefOutMsg)

# B-DOT: tamComm -> BdotControl -> MtbEffector
B_ref = 3.0e-5
J_min = min(hubI11, hubI22, hubI33)
kGainRef = 4.0 * np.pi * (1.0 + np.sin(oe.i)) * J_min / (P * B_ref ** 2)

kGain = kGainRef          # [A*m^2 / (T/s)];
bdotFilterTau = 0.5       # [s]

bdotControlObj = BdotControl(kGain=kGain, maxDipole=maxDipole, filterTau=bdotFilterTau)
bdotControlObj.tamInMsg.subscribeTo(tamCommObj.tamOutMsg)
scSim.AddModelToTask(fswTaskName, bdotControlObj)

mtbEff.mtbCmdInMsg.subscribeTo(bdotControlObj.dipoleCmdOutMsg)
mtbEff.mtbParamsInMsg.subscribeTo(mtbParamsInMsg)
mtbEff.magInMsg.subscribeTo(magModule.envOutMsgs[0])

#simulation
simulationTime = macros.sec2nano(5*P)

numDataPoints = 1500
samplingTime = simHelpers.samplingTime(simulationTime, simulationTimeStep, numDataPoints)

# Estado real
scStateLog = scObject.scStateOutMsg.recorder(samplingTime)
attNavLog = sNavObject.attOutMsg.recorder(samplingTime)
scSim.AddModelToTask(simTaskName, scStateLog)
scSim.AddModelToTask(simTaskName, attNavLog)

# Sensores e atuador
magLog = magModule.envOutMsgs[0].recorder(samplingTime)
tamLog = TAM.tamDataOutMsg.recorder(samplingTime)
imuLog = ImuSensor.sensorOutMsg.recorder(samplingTime)
mtbLog = mtbEff.logger(["torqueExternalPntB_B"], samplingTime)
bdotCmdLog = bdotControlObj.dipoleCmdOutMsg.recorder(samplingTime)

scSim.AddModelToTask(simTaskName, magLog)
scSim.AddModelToTask(tamTaskName, tamLog)
scSim.AddModelToTask(simTaskName, imuLog)
scSim.AddModelToTask(simTaskName, mtbLog)
scSim.AddModelToTask(fswTaskName, bdotCmdLog)

# Ambiente / perturbações
deterministicAtmoLog = atmo.envOutMsgs[0].recorder(samplingTime)
stochasticAtmoLog = atmoBridge.stochAtmoMsg.recorder(samplingTime)
solarLog = solarBridge.stochSolarMsg.recorder(samplingTime)
eclipseLog = eclipseObject.eclipseOutMsgs[0].recorder(samplingTime)
dragLog = hubDrag.logger(["forceExternal_B", "torqueExternalPntB_B"], samplingTime)
srpLog = SRPEffector.logger(["forceExternal_B", "torqueExternalPntB_B"], samplingTime)
ggLog = ggEff.logger(["torqueExternalPntB_B"], samplingTime)

scSim.AddModelToTask(simTaskName, deterministicAtmoLog)
scSim.AddModelToTask(simTaskName, stochasticAtmoLog)
scSim.AddModelToTask(simTaskName, solarLog)
scSim.AddModelToTask(simTaskName, eclipseLog)
scSim.AddModelToTask(simTaskName, dragLog)
scSim.AddModelToTask(simTaskName, srpLog)
scSim.AddModelToTask(simTaskName, ggLog)

#vizard
viz = vizSupport.enableUnityVisualization(
    scSim, 
    simTaskName, 
    scObject,
    saveFile="sim3U_CubeSat.bin")

viz.epochInMsg.subscribeTo(gravFactory.epochMsg)
viz.settings.ambient = 0.8

vizSupport.createCustomModel(
    viz,
    modelPath=r"C:\Users\azeve\Downloads\basilisk\cubesat3u_model\cimatelite.glb",   
    simBodiesToModify=["cimatelite"],
    scale=[7.0, 7.0, 7.0])  

# EXECUÇÃO
print(f"Período orbital P = {P:.1f} s | Ganho B-dot = {kGain:.3e} A*m^2/(T/s) | "
      f"Dipolo máx = {maxDipole} A*m^2")

scSim.InitializeSimulation()
scSim.ConfigureStopTime(simulationTime)
scSim.ExecuteSimulation()

# ==============================================================================
# PÓS-PROCESSAMENTO (cada registrador usa o seu próprio vetor de tempo)
# ==============================================================================
t_sc = scStateLog.times() * macros.NANO2SEC
t_imu = imuLog.times() * macros.NANO2SEC
t_cmd = bdotCmdLog.times() * macros.NANO2SEC
t_mtb = mtbLog.times() * macros.NANO2SEC

orb_sc = t_sc / P
orb_imu = t_imu / P
orb_cmd = t_cmd / P
orb_mtb = t_mtb / P

# Velocidade angular real e medida pela IMU [deg/s]
omega_true = np.array(scStateLog.omega_BN_B) * macros.R2D
omega_true_norm = np.linalg.norm(omega_true, axis=1)

omega_imu = np.array(imuLog.AngVelPlatform) * macros.R2D
omega_imu_norm = np.linalg.norm(omega_imu, axis=1)

# Comando e torque do MTB
dipole_cmd = np.array(bdotCmdLog.mtbDipoleCmds)[:, :3]
torque_mtb = np.array(mtbLog.torqueExternalPntB_B)
torque_mtb_norm = np.linalg.norm(torque_mtb, axis=1)

# Instante em que ||omega|| cruza o limiar do alarme
omegaAlarm_deg = 0.5


def firstBelow(t, y, thr):
    idx = np.where(y < thr)[0]
    return t[idx[0]] if idx.size else None


t_true_alarm = firstBelow(t_sc, omega_true_norm, omegaAlarm_deg)
t_imu_alarm = firstBelow(t_imu, omega_imu_norm, omegaAlarm_deg)

print("\n===== RESUMO DO DETUMBLING =====")
print(f"||omega|| inicial (real)   : {omega_true_norm[0]:.2f} deg/s")
print(f"||omega|| final   (real)   : {omega_true_norm[-1]:.3f} deg/s")
print(f"||omega|| final   (IMU)    : {omega_imu_norm[-1]:.3f} deg/s")
if t_true_alarm is not None:
    print(f"Real < {omegaAlarm_deg} deg/s em t = {t_true_alarm/60:.1f} min ({t_true_alarm/P:.2f} órbitas)")
else:
    print(f"Real NÃO atingiu {omegaAlarm_deg} deg/s")
if t_imu_alarm is not None:
    print(f"IMU  < {omegaAlarm_deg} deg/s em t = {t_imu_alarm/60:.1f} min")
else:
    print(f"IMU NUNCA atingiu {omegaAlarm_deg} deg/s (bias do giroscópio ~ "
          f"{np.linalg.norm(np.array(ImuSensor.senRotBias)) * macros.R2D:.1f} deg/s)")

# GRÁFICOS
plt.close("all")

# Figura 1: decaimento da velocidade angular
fig1, (ax1, ax1b) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax1.plot(orb_sc, omega_true[:, 0], label=r'$\omega_x$', color='tab:red', alpha=0.7)
ax1.plot(orb_sc, omega_true[:, 1], label=r'$\omega_y$', color='tab:green', alpha=0.7)
ax1.plot(orb_sc, omega_true[:, 2], label=r'$\omega_z$', color='tab:blue', alpha=0.7)
ax1.plot(orb_sc, omega_true_norm, label=r'$\|\omega\|$ real', color='black', linewidth=2.0)
ax1.set_title("Amortecimento da Velocidade Angular pelo B-dot")
ax1.set_ylabel("Velocidade Angular [deg/s]")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.6)

ax1b.semilogy(orb_sc, omega_true_norm, label=r'$\|\omega\|$ real', color='black', linewidth=2.0)
ax1b.semilogy(orb_imu, omega_imu_norm, label=r'$\|\omega\|$ medida (IMU MPU9250)',
              color='tab:purple', linewidth=1.2, alpha=0.8)
ax1b.axhline(omegaAlarm_deg, color='orange', linestyle='--',
             label=r'Alarme ($0.5^\circ/\mathrm{s}$)')
ax1b.set_xlabel("Tempo [Órbitas]")
ax1b.set_ylabel("Magnitude [deg/s]")
ax1b.legend(loc="upper right")
ax1b.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()

# Figura 2: dipolo comandado
fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(orb_cmd, dipole_cmd[:, 0], label=r'$m_x$', color='tab:red')
ax2.plot(orb_cmd, dipole_cmd[:, 1], label=r'$m_y$', color='tab:green')
ax2.plot(orb_cmd, dipole_cmd[:, 2], label=r'$m_z$', color='tab:blue')
ax2.axhline(maxDipole, color='black', linestyle='--', alpha=0.5,
            label=rf'Saturação ($\pm {maxDipole}\ \mathrm{{A\cdot m^2}}$)')
ax2.axhline(-maxDipole, color='black', linestyle='--', alpha=0.5)
ax2.set_title(r"Momento Dipolo Magnético Comandado ao MTB ($m_{cmd}$)")
ax2.set_xlabel("Tempo [Órbitas]")
ax2.set_ylabel(r"Dipolo [A$\cdot$m$^2$]")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Figura 3: torque gerado pelo MTB
fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.plot(orb_mtb, torque_mtb[:, 0] * 1e6, label=r'$\tau_x$', color='tab:red')
ax3.plot(orb_mtb, torque_mtb[:, 1] * 1e6, label=r'$\tau_y$', color='tab:green')
ax3.plot(orb_mtb, torque_mtb[:, 2] * 1e6, label=r'$\tau_z$', color='tab:blue')
ax3.plot(orb_mtb, torque_mtb_norm * 1e6, label=r'$\|\tau\|$', color='black', linewidth=1.2)
ax3.set_title(r"Torque Gerado pelo MTB ($\tau = m \times B$)")
ax3.set_xlabel("Tempo [Órbitas]")
ax3.set_ylabel(r"Torque [$\mu$N$\cdot$m]")
ax3.legend(loc="upper right")
ax3.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.show()