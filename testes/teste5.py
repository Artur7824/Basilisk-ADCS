#esse modelo inclui os sensores star tracker e giroscópio, fechando a malha de controle

import matplotlib.pyplot as plt
import numpy as np

from Basilisk.architecture import messaging, sysModel

from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    RigidBodyKinematics as rbk,  # Conversão MRP -> quatérnio (MRP2EP)
    orbitalMotion,
    simHelpers,                 
    simIncludeGravBody,
    vizSupport,
    fswSetupRW,
    simIncludeRW,
)
from Basilisk.utilities.supportDataTools.dataFetcher import get_path, DataFile

from Basilisk.simulation import (
    GravityGradientEffector,
    extForceTorque,
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
    reactionWheelStateEffector,
    motorVoltageInterface,
    MtbEffector,
    starTracker,
    boreAngCalc,
    imuSensor,
)

from Basilisk.fswAlgorithms import (
    attTrackingError,
    hillPoint, 
    mrpFeedback,
    inertial3D,
    rwMotorTorque,
    rwMotorVoltage,
    tamComm,
    mtbMomentumManagement,
)

#gemini + LM + claude ~ conexão do ruído estocástico com facetdrag
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
        # O objeto de estado é obtido no primeiro UpdateState, quando o
        # spacecraft já registrou todos os estados.
        self.stateObject = None
        # Escreve uma mensagem inicial para que a saída exista desde t = 0
        self.stochAtmoPayload = messaging.AtmoPropsMsgPayload()
        self.stochAtmoMsg.write(self.stochAtmoPayload, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        if self.stateObject is None:
            self.stateObject = self.scObject.dynManager.getStateObject(self.stateName)

        # delta_rho: estado escalar atual do processo de Ornstein-Uhlenbeck
        delta_rho = float(np.array(self.stateObject.getState()).flatten()[0])

        # Densidade nominal do MSIS [kg/m^3]
        nominalDensity = self.atmoModule.envOutMsgs[0].read().neutralDensity

        # rho_stoch = rho_nominal * (1 + delta_rho), limitada a valores não negativos
        self.stochAtmoPayload.neutralDensity = max(0.0, nominalDensity * (1.0 + delta_rho))

        self.stochAtmoMsg.write(self.stochAtmoPayload, CurrentSimNanos, self.moduleID)

#derivado diretamente do anterior
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


#configuração da simulação
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

#definição do satélite
scObject = spacecraft.Spacecraft()
scObject.ModelTag = "cimatelite"

hubMass = 4.0  # [kg]
scObject.hub.mHub = hubMass

# CubeSat 3U
hubWidthX = 0.1   # [m]
hubWidthY = 0.1   # [m]
hubHeightZ = 0.3  # [m]

hubAreaX = hubWidthY * hubHeightZ
hubAreaY = hubWidthX * hubHeightZ
hubAreaZ = hubWidthX * hubWidthY

hubHalfX = hubWidthX / 2.0   # 0.05 m
hubHalfY = hubWidthY / 2.0   # 0.05 m
hubHalfZ = hubHeightZ / 2.0  # 0.15 m

#inércia de um paralelepípedo retângulo sobre o centro de massa
hubI11 = hubMass / 12.0 * (hubWidthY ** 2 + hubHeightZ ** 2)
hubI22 = hubMass / 12.0 * (hubWidthX ** 2 + hubHeightZ ** 2)
hubI33 = hubMass / 12.0 * (hubWidthX ** 2 + hubWidthY ** 2)

I = [hubI11, 0.0, 0.0,
     0.0, hubI22, 0.0,
     0.0, 0.0, hubI33]

scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
scObject.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(I)

# Condições iniciais de atitude ~ sugestão do LM
scObject.hub.sigma_BNInit = [[0.8], [-1.2], [1.0]]  # MRP
scObject.hub.omega_BN_BInit = [
    [60.0 * macros.D2R],
    [40.0 * macros.D2R],
    [50.0 * macros.D2R],
]  # [rad/s]

#definição dos corpos celestes (Terra e Sol) e interface SPICE
gravFactory = simIncludeGravBody.gravBodyFactory()
bodies = gravFactory.createBodies('earth', 'sun')

planet = bodies['earth']
planet.isCentralBody = True
sun = bodies['sun']

# Esféricos harmônicos de 10° ordem
ggm03s_full_path = get_path(DataFile.LocalGravData.GGM03S)
planet.useSphericalHarmonicsGravityModel(str(ggm03s_full_path), 10)
mu = planet.mu

gravFactory.addBodiesTo(scObject)

# definições de órbita
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

# Interface SPICE
spiceObject = gravFactory.createSpiceInterface(
    time="2020 MAY 21 18:28:03 (UTC)",
    epochInMsg=True,
)
spiceObject.zeroBase = "Earth"
scSim.AddModelToTask(simTaskName, spiceObject, ModelPriority=100)  # roda primeiro

earthStateMsg = spiceObject.planetStateOutMsgs[0]  # Terra
sunStateMsg = spiceObject.planetStateOutMsgs[1]    # Sol

# Adiciona o satélite à tarefa principal
scSim.AddModelToTask(simTaskName, scObject)

# SOSRA é o integrador forte recomendado para ruído aditivo ~ scenarioStochasticDragSpacecraft
integrator = svIntegrators.svStochasticIntegratorSOSRA(scObject)
scObject.setIntegrator(integrator)

#Definição das RW ~ Hyperion RW210
v_nom = 3.3       # [V] Tensão nominal operacional (3.25V - 3.5V)
torque_max = 0.0001  # [N.m] Torque máximo

rwFactory = simIncludeRW.rwFactory()

varRWModel = messaging.BalancedWheels
#varRWModel = messaging.JitterSimple

RW1 = rwFactory.create('custom', [1, 0, 0], maxMomentum=0.006, Omega=0.0  # RPM inicial
                       , Omega_max = 15000.0 #RPM
                       , u_max = torque_max
                       , rWB_B=[0, 0, 0]  # meters
                       , RWModel=varRWModel)

RW2 = rwFactory.create('custom', [0, 1, 0], maxMomentum=0.006, Omega=0.0  # RPM inicial
                       , Omega_max = 15000.0 #RPM
                       , u_max = torque_max
                       , rWB_B=[0, 0, 0]  # meters
                       , RWModel=varRWModel)

RW3 = rwFactory.create('custom', [0, 0, 1], maxMomentum=0.006, Omega=0.0  # RPM inicial
                       , Omega_max = 15000.0 #RPM
                       , u_max = torque_max
                       , rWB_B=[0, 0, 0]  # meters
                       , RWModel=varRWModel)

numRW = rwFactory.getNumOfDevices()
rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
rwStateEffector.ModelTag = "RW_cluster"
rwFactory.addToSpacecraft(rwStateEffector.ModelTag, rwStateEffector, scObject)

scSim.AddModelToTask(simTaskName, rwStateEffector, 2)

gain_rw210 = torque_max / v_nom  # Ganho de conversão [N.m/V]

rwVoltageIO = motorVoltageInterface.MotorVoltageInterface()
rwVoltageIO.ModelTag = "rwVoltageInterface"
rwVoltageIO.setGains(np.array([gain_rw210] * numRW))

scSim.AddModelToTask(simTaskName, rwVoltageIO)

#Campo magnético WMM25
magModule = magneticFieldWMM.MagneticFieldWMM()
wmm_path = get_path(DataFile.MagneticFieldData.WMM)
magModule.configureWMMFile(str(wmm_path))
magModule.ModelTag = "WMM"

magModule.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, magModule)

magModule.epochInMsg.subscribeTo(gravFactory.epochMsg)

#3-axis magnetometer ~ MPU9250
TAM = magnetometer.Magnetometer()
TAM.ModelTag = "TAM_sensor"
TAM.scaleFactor = 1.0

TAM.senNoiseStd = [600e-9, 600e-9, 600e-9] # 0.6µT
TAM.senBias = [0, 0, -1e-6]  # copiado do example do basilisk
TAM.maxOutput = 4.8e-3   #  4800µT
TAM.minOutput = -4.8e-3  # -4800µT

tamCommObj = tamComm.tamComm()
tamCommObj.dcm_BS = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
tamCommObj.ModelTag = "tamComm"

TAM.stateInMsg.subscribeTo(scObject.scStateOutMsg)
TAM.magInMsg.subscribeTo(magModule.envOutMsgs[0])
tamCommObj.tamInMsg.subscribeTo(TAM.tamDataOutMsg)

scSim.AddModelToTask(tamTaskName, TAM)
scSim.AddModelToTask(simTaskName, tamCommObj)

#Magnetorquer ~ Hyperion HT-MTQ200
mtbEff = MtbEffector.MtbEffector()
mtbEff.ModelTag = "MtbEff"
scObject.addDynamicEffector(mtbEff)
scSim.AddModelToTask(simTaskName, mtbEff)

mtbMomentumManagementObj = mtbMomentumManagement.mtbMomentumManagement()
mtbMomentumManagementObj.wheelSpeedBiases = [1000. * macros.rpm2radsec, 1000. * macros.rpm2radsec,
                                             1000. * macros.rpm2radsec]
mtbMomentumManagementObj.cGain = 0.003
mtbMomentumManagementObj.ModelTag = "mtbMomentumManagement"
scSim.AddModelToTask(simTaskName, mtbMomentumManagementObj)

mtbConfigParams = messaging.MTBArrayConfigMsgPayload()
mtbConfigParams.numMTB = 3
maxDipole = 1.0
mtbConfigParams.GtMatrix_B =[1.0, 0., 0.,
                             0., 1.0, 0.,
                             0., 0., 1.0]

mtbConfigParams.maxMtbDipoles = [maxDipole]*mtbConfigParams.numMTB
mtbParamsInMsg = messaging.MTBArrayConfigMsg().write(mtbConfigParams)

#Star Tracker
ST = starTracker.StarTracker()
ST.ModelTag = "StarTracker"

dcm_CB_matrix = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0],
    [0.0,  1.0,  0.0]])

ST.dcm_CB = dcm_CB_matrix.tolist()

# ST-200
arcsec2rad = (np.pi / 180.0) / 3600.0
stdCorrectionFactor = 1.5  # Fator de ajuste "padrão"

# Desvio padrão 1-sigma
sigma_pitch_yaw = (30.0 / 3.0) * arcsec2rad   # ~4.85e-5 rad (Pitch e Yaw)
sigma_roll      = (200.0 / 3.0) * arcsec2rad  # ~3.23e-4 rad (Roll)

P_x = (sigma_pitch_yaw * stdCorrectionFactor) ** 2
P_y = (sigma_pitch_yaw * stdCorrectionFactor) ** 2
P_z = (sigma_roll * stdCorrectionFactor) ** 2

ST.PMatrix = [
    [P_x, 0.0, 0.0],
    [0.0, P_y, 0.0],
    [0.0, 0.0, P_z]]

walkBound_transversal = 60.0 * arcsec2rad   # ~2.91e-4 rad
walkBound_boresight   = 400.0 * arcsec2rad  # ~1.94e-3 rad

ST.walkBounds = [
    [walkBound_transversal], 
    [walkBound_transversal], 
    [walkBound_boresight]]

ST.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
scSim.AddModelToTask(stTaskName, ST)

#Cone de visão do Star Tracker

#Sol
stSunBAC = boreAngCalc.BoreAngCalc()
stSunBAC.ModelTag = "starTrackerSunBoresight"
stSunBAC.boreVec_B = [0., -1., 0.] 
stSunBAC.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
stSunBAC.celBodyInMsg.subscribeTo(sunStateMsg)
scSim.AddModelToTask(simTaskName, stSunBAC)

# terra
stEarthBAC = boreAngCalc.BoreAngCalc()
stEarthBAC.ModelTag = "starTrackerEarthBoresight"
stEarthBAC.boreVec_B = [0., -1., 0.] 
stEarthBAC.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
stEarthBAC.celBodyInMsg.subscribeTo(earthStateMsg)
scSim.AddModelToTask(simTaskName, stEarthBAC)

#sensor IMU (gyro + acel) ~ MPU9250
ImuSensor = imuSensor.ImuSensor()
ImuSensor.ModelTag = "imusensor"
ImuSensor.sensorPos_B = [0.0, 0.0, 0.0]

yaw = 0.0 # [rad]
pitch = 0.0  # [rad]
roll = 0.0 # [rad]

ImuSensor.setBodyToPlatformDCM(yaw, pitch, roll) 

ImuSensor.senRotMax = 2000.0 * macros.D2R

#bias máximo de 5°
gyro_bias = 5.0 * macros.D2R #bias máximo
ImuSensor.senRotBias = [gyro_bias,-gyro_bias*0.7,gyro_bias/2]

gyro_std = 0.1 * macros.D2R 
ImuSensor.PMatrixGyro = [[gyro_std, 0.0,      0.0],
                         [0.0,      gyro_std, 0.0],
                         [0.0,      0.0,      gyro_std]]

walk_gyro = 30.0 * macros.D2R 
ImuSensor.setErrorBoundsGyro = np.array(walk_gyro)

g_accel = 9.80665
ImuSensor.senTransMax = 16.0 * g_accel  

#bias máximo de 60 e 80mg
accel_bias_xy = 0.060 * g_accel
accel_bias_z = 0.080 * g_accel
ImuSensor.senTransBias = [-accel_bias_xy,accel_bias_xy/2,accel_bias_z]

accel_std = 0.008 * g_accel  # ~0.07845 m/s^2
ImuSensor.PMatrixAccel = [[accel_std, 0.0,       0.0],
                          [0.0,       accel_std, 0.0],
                          [0.0,       0.0,       accel_std]]

walk_accel = 0.0015 * g_accel
ImuSensor.setErrorBoundsAccel = [walk_accel, walk_accel, walk_accel]

ImuSensor.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, ImuSensor)

# Atmosfera MSIS
atmo = msisAtmosphere.MsisAtmosphere()
atmo.ModelTag = "MsisAtmo"

# Índices de atividade solar (F10.7 e Ap)
ap = 8
f107 = 110
#atualização futura - usar dados reais de clima espacial
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
# Prioridade acima da ponte: a ponte precisa ler a densidade já atualizada do MSIS
scSim.AddModelToTask(simTaskName, atmo, ModelPriority=90)

# Variação estocástica da densidade
stochasticNoise = meanRevertingNoiseStateEffector.MeanRevertingNoiseStateEffector()
stochasticNoise.ModelTag = "StochasticExpAtmo"
stochasticNoise.setStationaryStd(0.15)       # desvio padrão estacionário de delta_rho
stochasticNoise.setTimeConstant(10 * 60.0)  # [s]
scObject.addStateEffector(stochasticNoise)

atmoBridge = StochasticAtmoBridge(scObject, atmo, stochasticNoise.getStateName())
scSim.AddModelToTask(simTaskName, atmoBridge, ModelPriority=80)

# modelagem do vento ~ zerowind model
windModel = zeroWindModel.ZeroWindModel()
windModel.ModelTag = "ZeroWind"
windModel.planetPosInMsg.subscribeTo(earthStateMsg)
windModel.epochInMsg.subscribeTo(gravFactory.epochMsg)
windModel.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, windModel)

#modelagem SRP
SRPEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
SRPEffector.ModelTag = "SRPEffector"
SRPEffector.setNumFacets(6) 

#ruído estocástico no fluxo solar
stochasticSolarNoise = meanRevertingNoiseStateEffector.MeanRevertingNoiseStateEffector()
stochasticSolarNoise.ModelTag = "StochasticSolarNoise"
stochasticSolarNoise.setStationaryStd(0.01)          # Desvio padrão de ~1% no fluxo solar
stochasticSolarNoise.setTimeConstant(10.0 * 60.0)     
scObject.addStateEffector(stochasticSolarNoise)

solarBridge = StochasticSolarBridge(
    scObject=scObject, 
    sunStateMsg=sunStateMsg, 
    stateName=stochasticSolarNoise.getStateName(),
    nominalSolarFlux=1361.0  # Constante solar nominal [W/m^2]
)
scSim.AddModelToTask(simTaskName, solarBridge, ModelPriority=85)

#Facets: Aerodrag e SRP
dragCoeff = 2.2
sigSpec = 0.068 #referente aos materiais da placa solar ~ GaInP/GaAs/Ge on Ge substrate
sigDiff = 0.012 #idem

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

            # Aerodrag (4 parâmetros)
            facetDrag.addFacet(areas[axis], dragCoeff,
                               np.array(normal_list), np.array(location_list))

            # SRP (7 parâmetros)
            facetSRP.addFacet(
                areas[axis],    # 1. Área [m^2]
                dcm_identity,   # 2. Matriz de orientação da faceta [3x3]
                normal_list,    # 3. Normal da faceta [1x3]
                rot_list,       # 4. Eixo de rotação [1x3]
                location_list,  # 5. Localização / centro de pressão [1x3]
                sigDiff,        # 6. Coeficiente difuso
                sigSpec,        # 7. Coeficiente especular
            )


addCubesatFacets(hubDrag, SRPEffector)

# Conexões do arrasto
hubDrag.atmoDensInMsg.subscribeTo(atmoBridge.stochAtmoMsg)
hubDrag.windVelInMsg.subscribeTo(windModel.envOutMsgs[0])
scObject.addDynamicEffector(hubDrag)
scSim.AddModelToTask(simTaskName, hubDrag)

#definição do eclipse
eclipseObject = eclipse.Eclipse()
eclipseObject.ModelTag = "EarthEclipse"

eclipseObject.addSpacecraftToModel(scObject.scStateOutMsg)
eclipseObject.addPlanetToModel(earthStateMsg)  # corpo eclipsante: Terra (SPICE [0])
eclipseObject.sunInMsg.subscribeTo(sunStateMsg)  # fonte luminosa: Sol (SPICE [1])
scSim.AddModelToTask(simTaskName, eclipseObject)

# Conexões do SRP
SRPEffector.sunEclipseInMsg.subscribeTo(eclipseObject.eclipseOutMsgs[0])
SRPEffector.sunInMsg.subscribeTo(sunStateMsg)
if hasattr(SRPEffector, "solarFluxInMsg"):
    SRPEffector.solarFluxInMsg.subscribeTo(solarBridge.stochSolarMsg)

scObject.addDynamicEffector(SRPEffector)
scSim.AddModelToTask(simTaskName, SRPEffector, ModelPriority=70)

#Gradiente de gravidade
ggEff = GravityGradientEffector.GravityGradientEffector()
ggEff.ModelTag = scObject.ModelTag
ggEff.addPlanetName(planet.planetName)
scObject.addDynamicEffector(ggEff)
scSim.AddModelToTask(simTaskName, ggEff)

#definição dos sensores de atitude
sNavObject = simpleNav.SimpleNav()
sNavObject.ModelTag = "SimpleNavigation"
scSim.AddModelToTask(fswTaskName, sNavObject)

# setup hillPoint guidance module
attGuidance = hillPoint.hillPoint()
attGuidance.ModelTag = "hillPoint"
scSim.AddModelToTask(fswTaskName, attGuidance)

# setup the attitude tracking error evaluation module
attError = attTrackingError.attTrackingError()
attError.ModelTag = "attError"
scSim.AddModelToTask(fswTaskName, attError)

attError.sigma_R0R = [0, 1, 0]

# Conecta o sensor de navegação aos estados reais do satélite
sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

attGuidance.transNavInMsg.subscribeTo(sNavObject.transOutMsg)

# Conecta o calculador de erro de atitude ao sensor e à referência Nadir
attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
attError.attRefInMsg.subscribeTo(attGuidance.attRefOutMsg)

# setup the MRP Feedback control module
mrpControl = mrpFeedback.mrpFeedback()
mrpControl.ModelTag = "mrpFeedback"
scSim.AddModelToTask(fswTaskName, mrpControl)

mrpControl.K = 0.001    # Proporcional
mrpControl.P = 0.015    # Derivativo 
mrpControl.Ki = -1.0    # Integral 

fswRwParamMsg = rwFactory.getConfigMessage()
vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B=I)
vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

#links
mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
mrpControl.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)

# add module that maps the Lr control torque into the RW motor torques
rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
rwMotorTorqueObj.ModelTag = "rwMotorTorque"
scSim.AddModelToTask(fswTaskName, rwMotorTorqueObj)

#links
rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)

# Make the RW control all three body axes
controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
rwMotorTorqueObj.controlAxes_B = controlAxes_B

fswRWVoltage = rwMotorVoltage.rwMotorVoltage()
fswRWVoltage.ModelTag = "rwMotorVoltage"

# Add test module to runtime call list
scSim.AddModelToTask(fswTaskName, fswRWVoltage)

fswRWVoltage.VMin = -3.5
fswRWVoltage.VMax = 3.5

#links MTB +RW
mtbMomentumManagementObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
mtbMomentumManagementObj.mtbParamsInMsg.subscribeTo(mtbParamsInMsg)
mtbMomentumManagementObj.tamSensorBodyInMsg.subscribeTo(tamCommObj.tamOutMsg)
mtbMomentumManagementObj.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)
mtbMomentumManagementObj.rwMotorTorqueInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)

fswRWVoltage.torqueInMsg.subscribeTo(mtbMomentumManagementObj.rwMotorTorqueOutMsg)
fswRWVoltage.rwParamsInMsg.subscribeTo(fswRwParamMsg)
rwVoltageIO.motorVoltageInMsg.subscribeTo(fswRWVoltage.voltageOutMsg)
rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwVoltageIO.motorTorqueOutMsg)

mtbEff.mtbCmdInMsg.subscribeTo(mtbMomentumManagementObj.mtbCmdOutMsg)
mtbEff.mtbParamsInMsg.subscribeTo(mtbParamsInMsg)
mtbEff.magInMsg.subscribeTo(magModule.envOutMsgs[0])

# set the simulation time
n = np.sqrt(mu / oe.a / oe.a / oe.a)
P = 2.0 * np.pi / n

simulationTime = macros.sec2nano(1*P)

#full gemini
# 1. CONFIGURAÇÃO E REGISTRO DE DADOS 
numDataPoints = 1000
samplingTime = simHelpers.samplingTime(simulationTime, simulationTimeStep, numDataPoints)

# Registradores de Navegação e Atitude
snLog = sNavObject.transOutMsg.recorder(samplingTime)
attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
attNavLog = sNavObject.attOutMsg.recorder(samplingTime)

scSim.AddModelToTask(simTaskName, snLog)
scSim.AddModelToTask(simTaskName, attNavLog)
scSim.AddModelToTask(simTaskName, attErrorLog)

# Registradores de Ambiente, Pontes Estocásticas e Eclipse
deterministicAtmoLog = atmo.envOutMsgs[0].recorder(samplingTime)
stochasticAtmoLog = atmoBridge.stochAtmoMsg.recorder(samplingTime)
solarLog = solarBridge.stochSolarMsg.recorder(samplingTime)
eclipseLog = eclipseObject.eclipseOutMsgs[0].recorder(samplingTime)
magLog = magModule.envOutMsgs[0].recorder(samplingTime)
tamLog = TAM.tamDataOutMsg.recorder(samplingTime)
mtbLog = mtbEff.logger(["torqueExternalPntB_B"], samplingTime)
stLog = ST.sensorOutMsg.recorder(stTimeStep) 
stSunBACLog = stSunBAC.angOutMsg.recorder(samplingTime)
stEarthBACLog = stEarthBAC.angOutMsg.recorder(samplingTime)
imuLog = ImuSensor.sensorOutMsg.recorder(samplingTime)

scSim.AddModelToTask(simTaskName, deterministicAtmoLog)
scSim.AddModelToTask(simTaskName, stochasticAtmoLog)
scSim.AddModelToTask(simTaskName, solarLog)
scSim.AddModelToTask(simTaskName, eclipseLog)
scSim.AddModelToTask(simTaskName, magLog)
scSim.AddModelToTask(tamTaskName, tamLog)
scSim.AddModelToTask(simTaskName, mtbLog)
scSim.AddModelToTask(stTaskName, stLog)
scSim.AddModelToTask(simTaskName, stSunBACLog)
scSim.AddModelToTask(simTaskName, stEarthBACLog)
scSim.AddModelToTask(simTaskName, imuLog)

# Registradores Únicos para os Efetores Dinâmicos
dragLog = hubDrag.logger(["forceExternal_B", "torqueExternalPntB_B"], samplingTime)
srpLog = SRPEffector.logger(["forceExternal_B", "torqueExternalPntB_B"], samplingTime)
ggLog = ggEff.logger(["torqueExternalPntB_B"], samplingTime)

scSim.AddModelToTask(simTaskName, dragLog)
scSim.AddModelToTask(simTaskName, srpLog)
scSim.AddModelToTask(simTaskName, ggLog)

# Registradores das RW
rwSpeedLog = rwStateEffector.rwSpeedOutMsg.recorder(samplingTime)
scSim.AddModelToTask(simTaskName, rwSpeedLog)

rwCmdTorqueLog = rwMotorTorqueObj.rwMotorTorqueOutMsg.recorder(samplingTime)
scSim.AddModelToTask(fswTaskName, rwCmdTorqueLog)

rwVoltLog = fswRWVoltage.voltageOutMsg.recorder(samplingTime)
scSim.AddModelToTask(fswTaskName, rwVoltLog)

# 2. DEFINIÇÃO DO ARQUICO DE VISUALIZAÇÃO
viz = vizSupport.enableUnityVisualization(
    scSim, 
    simTaskName, 
    scObject,
    saveFile="sim3U_CubeSat.bin",
    rwEffectorList=rwStateEffector)

viz.epochInMsg.subscribeTo(gravFactory.epochMsg)
viz.settings.ambient = 0.8

import os

diretorio_atual = os.path.dirname(os.path.abspath(__file__))

# Certifique-se de mover a pasta 'vizard' para junto do seu script .py
model_path = os.path.join(diretorio_atual, "vizard", "cimatelite.glb")

# Executa a função com o caminho universal
vizSupport.createCustomModel(
    viz, 
    modelPath=model_path, 
    simBodiesToModify=["cimatelite"], 
    scale=[7.0, 7.0, 7.0]
)

starTrackerFov_deg = 40 #angulo de exclusão solar

vizSupport.createConeInOut(
    viz, 
    toBodyName='sun',                               # Referência astronômica
    coneColor='g',                                  # Verde (Green)
    normalVector_B=[0.0, -1.0, 0.0],                # Saída na Face Lateral 1 (-Y)
    incidenceAngle=20 * macros.D2R,                 # Abertura de 20 graus
    isKeepIn=True,                                  # True = Keep-In (Cone de visada nominal)
    coneHeight=5.0,                                 # Comprimento visual
    coneName='ST200_FOV_Verde'
)

vizSupport.createConeInOut(
    viz, 
    toBodyName='sun', 
    coneColor='r',                                  # Vermelho (Red)
    normalVector_B=[0.0, -1.0, 0.0], 
    incidenceAngle=20 * macros.D2R, 
    isKeepIn=False,                                 # False = Keep-Out (Área de violação)
    coneHeight=5.0, 
    coneName='ST200_Sun_KeepOut'
)

# 3. EXECUÇÃO DA SIMULAÇÃO
scSim.InitializeSimulation()
scSim.ConfigureStopTime(simulationTime)
scSim.ExecuteSimulation()

# ------------------------------------------------------------------------------
# 4. EXTRAÇÃO E PROCESSAMENTO DOS DADOS (Com Tempos Independentes por Task)
# ------------------------------------------------------------------------------

# Vetores temporais individuais por registrador (Evita erro de dimensão)
timeOrbits_sim = (snLog.times() * macros.NANO2SEC) / P
timeOrbits_fsw = (rwVoltLog.times() * macros.NANO2SEC) / P
timeOrbits_tam = (tamLog.times() * macros.NANO2SEC) / P
timeOrbits_st  = (stLog.times() * macros.NANO2SEC) / P
timeOrbits_imu = (imuLog.times() * macros.NANO2SEC) / P

# Variáveis globais baseadas na simTask
posData_global = snLog.r_BN_N
attErrSigma_global = attErrorLog.sigma_BR
attErrOmega_global = attErrorLog.omega_BR_B
quatErr_global = np.array([rbk.MRP2EP(s) for s in attErrSigma_global])

# Dados do FSW e Motores
dataOmegaRW = rwSpeedLog.wheelSpeeds / macros.RPM
dataVoltRW = rwVoltLog.voltage
dataTorqueMTB = mtbLog.torqueExternalPntB_B

# Magnitudes de Força [N] e Torque [N*m]
magForceDrag = np.linalg.norm(dragLog.forceExternal_B, axis=1)
magForceSRP  = np.linalg.norm(srpLog.forceExternal_B, axis=1)
magTorqueDrag = np.linalg.norm(dragLog.torqueExternalPntB_B, axis=1)
magTorqueSRP  = np.linalg.norm(srpLog.torqueExternalPntB_B, axis=1)
magTorqueGG   = np.linalg.norm(ggLog.torqueExternalPntB_B, axis=1)

# Validação do Star Tracker
sun_angles_deg = stSunBACLog.missAngle * macros.R2D
earth_angles_deg = stEarthBACLog.missAngle * macros.R2D

sun_angles_interp = np.interp(stLog.times(), stSunBACLog.times(), sun_angles_deg)
earth_angles_interp = np.interp(stLog.times(), stEarthBACLog.times(), earth_angles_deg)

fov_threshold_deg = 40.0
st_blind_mask = (sun_angles_interp <= fov_threshold_deg) | (earth_angles_interp <= fov_threshold_deg)

sigma_BN_real = attNavLog.sigma_BN
quat_BN_real = np.array([rbk.MRP2EP(s) for s in sigma_BN_real])
quat_CN_measured = stLog.qInrtl2Case

quat_BC = rbk.C2EP(dcm_CB_matrix.T)
quat_BN_measured = np.zeros_like(quat_CN_measured)
for i in range(len(quat_CN_measured)):
    quat_BN_measured[i] = rbk.addEP(quat_CN_measured[i], quat_BC)

quat_real_interp = np.zeros_like(quat_BN_measured)
for i in range(4):
    quat_real_interp[:, i] = np.interp(stLog.times(), snLog.times(), quat_BN_real[:, i])

error_arcsec = np.zeros((len(quat_BN_measured), 3))
for i in range(len(quat_BN_measured)):
    q_m = quat_BN_measured[i]
    q_r = quat_real_interp[i]
    if np.dot(q_m, q_r) < 0.0:
        q_m = -q_m
    dq = rbk.subEP(q_m, q_r)
    error_arcsec[i] = 2.0 * dq[1:] * (180.0 / np.pi) * 3600.0

quat_BN_measured_masked = np.copy(quat_BN_measured)
error_arcsec_masked = np.copy(error_arcsec)
quat_BN_measured_masked[st_blind_mask] = np.nan
error_arcsec_masked[st_blind_mask] = np.nan

# Processamento da IMU (Giroscópio + Acelerômetro)
omega_BN_real_deg = np.array(attNavLog.omega_BN_B) * macros.R2D
omega_BN_measured_deg = np.array(imuLog.AngVelPlatform) * macros.R2D

omega_real_interp = np.zeros_like(omega_BN_measured_deg)
for i in range(3):
    omega_real_interp[:, i] = np.interp(imuLog.times(), attNavLog.times(), omega_BN_real_deg[:, i])

gyro_error_deg = omega_BN_measured_deg - omega_real_interp

dt_imu = macros.NANO2SEC * simulationTimeStep
accel_measured_m_s2 = np.array(imuLog.DVFramePlatform) / dt_imu
accel_measured_g = accel_measured_m_s2 / g_accel

force_total_B = dragLog.forceExternal_B + srpLog.forceExternal_B
accel_real_m_s2 = force_total_B / hubMass
accel_real_g = accel_real_m_s2 / g_accel

accel_real_interp_g = np.zeros_like(accel_measured_g)
for i in range(3):
    accel_real_interp_g[:, i] = np.interp(imuLog.times(), dragLog.times(), accel_real_g[:, i])

accel_error_mg = (accel_measured_g - accel_real_interp_g) * 1000.0

magData = magLog.magField_N
tamData = tamLog.tam_S

# ==============================================================================
# 5. PLOTAGEM ESTRUTURADA E ORGANIZADA (Corrigido com eixos temporais dedicados)
# ==============================================================================
plt.close("all")

# Alias para dados gerados na simTask (1 kHz)
timeOrbits = timeOrbits_sim

# ------------------------------------------------------------------------------
# FIGURA 1: Erro de Atitude (Quatérnio de Erro em Relação ao Nadir)
# ------------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(timeOrbits, quatErr_global[:, 0], label=r'$q_0$ (Escalar)', color='black', linewidth=1.2)
ax1.plot(timeOrbits, quatErr_global[:, 1], label=r'$q_1$ (Eixo $x$)', color='tab:red')
ax1.plot(timeOrbits, quatErr_global[:, 2], label=r'$q_2$ (Eixo $y$)', color='tab:green')
ax1.plot(timeOrbits, quatErr_global[:, 3], label=r'$q_3$ (Eixo $z$)', color='tab:blue')
ax1.set_title(r"Evolução do Quatérnio de Erro em Relação ao Nadir ($q_{B/R}$)")
ax1.set_xlabel("Tempo [Órbitas]")
ax1.set_ylabel("Componentes $q$")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 2: Erro de Velocidade Angular
# ------------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 4))
omega_deg = attErrOmega_global * macros.R2D
ax2.plot(timeOrbits, omega_deg[:, 0], label=r'$\omega_{x}$', color='tab:red')
ax2.plot(timeOrbits, omega_deg[:, 1], label=r'$\omega_{y}$', color='tab:green')
ax2.plot(timeOrbits, omega_deg[:, 2], label=r'$\omega_{z}$', color='tab:blue')
ax2.set_title(r"Erro de Velocidade Angular ($\omega_{B/R}$)")
ax2.set_xlabel("Tempo [Órbitas]")
ax2.set_ylabel("Velocidade Angular [deg/s]")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 3: Torque de Gradiente de Gravidade
# ------------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.semilogy(timeOrbits[1:], magTorqueGG[1:], color='tab:blue', linewidth=1.2)
ax3.set_title(r"Magnitude do Torque de Gradiente de Gravidade ($|L_{GG}|$)")
ax3.set_xlabel("Tempo [Órbitas]")
ax3.set_ylabel("Torque [N·m]")
ax3.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 4: Força Aerodinâmica (Arrasto)
# ------------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(9, 4))
ax4.semilogy(timeOrbits[1:], magForceDrag[1:], color='tab:blue', linewidth=1.2)
ax4.set_title(r"Magnitude da Força de Arrasto Aerodinâmico ($|F_{drag}|$)")
ax4.set_xlabel("Tempo [Órbitas]")
ax4.set_ylabel("Força [N]")
ax4.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 5: Torque Aerodinâmico (Arrasto)
# ------------------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(9, 4))
ax5.semilogy(timeOrbits[1:], magTorqueDrag[1:], color='tab:red', linewidth=1.2)
ax5.set_title(r"Magnitude do Torque de Arrasto Aerodinâmico ($|L_{drag}|$)")
ax5.set_xlabel("Tempo [Órbitas]")
ax5.set_ylabel("Torque [N·m]")
ax5.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 6: Força SRP (Pressão de Radiação Solar)
# ------------------------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(9, 4))
ax6.semilogy(timeOrbits[1:], magForceSRP[1:], color='tab:orange', linewidth=1.2)
ax6.set_title(r"Magnitude da Força de Pressão de Radiação Solar ($|F_{SRP}|$)")
ax6.set_xlabel("Tempo [Órbitas]")
ax6.set_ylabel("Força [N]")
ax6.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 7: Torque SRP (Pressão de Radiação Solar)
# ------------------------------------------------------------------------------
fig7, ax7 = plt.subplots(figsize=(9, 4))
ax7.semilogy(timeOrbits[1:], magTorqueSRP[1:], color='tab:orange', linewidth=1.2)
ax7.set_title(r"Magnitude do Torque de Pressão de Radiação Solar ($|L_{SRP}|$)")
ax7.set_xlabel("Tempo [Órbitas]")
ax7.set_ylabel("Torque [N·m]")
ax7.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 8: Campo Magnético Medido (TAM MPU9250)
# ------------------------------------------------------------------------------
fig8, ax8 = plt.subplots(figsize=(9, 4))
ax8.plot(timeOrbits_tam, tamData[:, 0] * 1e9, label=r'$B_{B,x}$', color='tab:red')
ax8.plot(timeOrbits_tam, tamData[:, 1] * 1e9, label=r'$B_{B,y}$', color='tab:green')
ax8.plot(timeOrbits_tam, tamData[:, 2] * 1e9, label=r'$B_{B,z}$', color='tab:blue')
ax8.set_title("Vetor de Campo Magnético Medido no Referencial do Satélite (TAM)")
ax8.set_xlabel("Tempo [Órbitas]")
ax8.set_ylabel("Campo Magnético [nT]")
ax8.legend(loc="upper right")
ax8.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 9: Velocidade Angular das Rodas de Reação (RPM)
# ------------------------------------------------------------------------------
fig9, ax9 = plt.subplots(figsize=(9, 4.5))
ax9.plot(timeOrbits, dataOmegaRW[:, 0], label=r'$\Omega_1$ (Eixo $x$)', color='tab:red', linewidth=1.2)
ax9.plot(timeOrbits, dataOmegaRW[:, 1], label=r'$\Omega_2$ (Eixo $y$)', color='tab:green', linewidth=1.2)
ax9.plot(timeOrbits, dataOmegaRW[:, 2], label=r'$\Omega_3$ (Eixo $z$)', color='tab:blue', linewidth=1.2)
ax9.axhline(15000.0, color='black', linestyle='--', alpha=0.7, label=r'Saturação ($\pm 15.000$ RPM)')
ax9.axhline(-15000.0, color='black', linestyle='--', alpha=0.7)
ax9.set_title("Velocidade Angular das Rodas de Reação Hyperion RW210")
ax9.set_xlabel("Tempo [Órbitas]")
ax9.set_ylabel("Velocidade [RPM]")
ax9.legend(loc="upper right")
ax9.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 10: Sinal de Tensão Comandado às RWs
# ------------------------------------------------------------------------------
fig10, ax10 = plt.subplots(figsize=(9, 4.5))
ax10.plot(timeOrbits_fsw, dataVoltRW[:, 0], label=r'$V_1$', color='tab:red', linewidth=1.2)
ax10.plot(timeOrbits_fsw, dataVoltRW[:, 1], label=r'$V_2$', color='tab:green', linewidth=1.2)
ax10.plot(timeOrbits_fsw, dataVoltRW[:, 2], label=r'$V_3$', color='tab:blue', linewidth=1.2)
ax10.axhline(3.5, color='black', linestyle='--', alpha=0.7, label=r'Limite de Tensão ($\pm 3.5$ V)')
ax10.axhline(-3.5, color='black', linestyle='--', alpha=0.7)
ax10.set_title("Sinal de Tensão Aplicado aos Motores (FSW -> Hardware)")
ax10.set_xlabel("Tempo [Órbitas]")
ax10.set_ylabel("Tensão [V]")
ax10.legend(loc="upper right")
ax10.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 11: Torque Magnético Gerado pelo MTB (Hyperion HT-MTQ200)
# ------------------------------------------------------------------------------
fig11, ax11 = plt.subplots(figsize=(9, 4.5))
ax11.plot(timeOrbits, dataTorqueMTB[:, 0] * 1e6, label=r'$\tau_{MTB,x}$', color='tab:red', linewidth=1.2)
ax11.plot(timeOrbits, dataTorqueMTB[:, 1] * 1e6, label=r'$\tau_{MTB,y}$', color='tab:green', linewidth=1.2)
ax11.plot(timeOrbits, dataTorqueMTB[:, 2] * 1e6, label=r'$\tau_{MTB,z}$', color='tab:blue', linewidth=1.2)
ax11.set_title("Torque Magnético Gerado pelo MTB Hyperion HT-MTQ200")
ax11.set_xlabel("Tempo [Órbitas]")
ax11.set_ylabel(r"Torque [$\mu\text{N}\cdot\text{m}$]")
ax11.legend(loc="upper right")
ax11.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 12: Comparação Quatérnio Real vs. Medido pelo Star Tracker
# ------------------------------------------------------------------------------
fig12, ax12 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
labels = [r'$q_0$ (Escalar)', r'$q_1$ (Eixo $x$)', r'$q_2$ (Eixo $y$)', r'$q_3$ (Eixo $z$)']
colors = ['black', 'tab:red', 'tab:green', 'tab:blue']

for i in range(4):
    ax12[i].plot(timeOrbits, quat_BN_real[:, i], label='Real ($q_{B/N}$)', color='black', alpha=0.7, linewidth=1.5)
    ax12[i].plot(timeOrbits_st, quat_BN_measured_masked[:, i], label='Medido ST ($q_{B/N}$)', color=colors[i], linestyle='--', linewidth=1.2)
    
    # Destaca em vermelho vertical os instantes onde o Star Tracker está desativado/ofuscado
    ax12[i].fill_between(
        timeOrbits_st, 
        0, 1, 
        where=st_blind_mask, 
        color='red', 
        alpha=0.35, 
        transform=ax12[i].get_xaxis_transform(),
        label='ST inativo' if i == 0 else ""
    )
    
    ax12[i].set_ylabel(labels[i])
    ax12[i].grid(True, linestyle="--", alpha=0.5)
    if i == 0:
        ax12[i].legend(loc="upper right", ncol=3)

ax12[3].set_xlabel("Tempo [Órbitas]")
fig12.suptitle("Atitude: Real vs. Medida pelo Star Tracker", fontsize=12)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 13: Erro de Medição do Star Tracker (arcsec)
# ------------------------------------------------------------------------------
fig13, ax13 = plt.subplots(figsize=(9, 4))
ax13.plot(timeOrbits_st, error_arcsec_masked[:, 0], label=r'Erro $x$', color='tab:red', alpha=0.8)
ax13.plot(timeOrbits_st, error_arcsec_masked[:, 1], label=r'Erro $y$', color='tab:green', alpha=0.8)
ax13.plot(timeOrbits_st, error_arcsec_masked[:, 2], label=r'Erro $z$', color='tab:blue', alpha=0.8)

# Adiciona a faixa vermelha vertical onde o sensor está inativo
ax13.fill_between(
    timeOrbits_st, 
    0, 1, 
    where=st_blind_mask, 
    color='red', 
    alpha=0.35, 
    transform=ax13.get_xaxis_transform(),
    label='ST Inativo'
)

ax13.set_title("Ruído/Erro do Star Tracker")
ax13.set_xlabel("Tempo [Órbitas]")
ax13.set_ylabel("Erro [arcsec]")
ax13.legend(loc="upper right")
ax13.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 14: Comparação Velocidade Angular Real vs. Medida pelo Giroscópio (IMU MPU9250)
# ------------------------------------------------------------------------------
fig14, ax14 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
labels_w = [r'$\omega_x$ (Eixo $x$)', r'$\omega_y$ (Eixo $y$)', r'$\omega_z$ (Eixo $z$)']
colors_w = ['tab:red', 'tab:green', 'tab:blue']

for i in range(3):
    ax14[i].plot(timeOrbits, omega_BN_real_deg[:, i], label=r'Real ($\omega_{B/N}$)', color='black', alpha=0.7, linewidth=1.5)
    ax14[i].plot(timeOrbits_imu, omega_BN_measured_deg[:, i], label=r'Medido IMU ($\omega_{P}$)', color=colors_w[i], linestyle='--', alpha=0.8, linewidth=1.0)
    ax14[i].set_ylabel(f"{labels_w[i]}\n[deg/s]")
    ax14[i].grid(True, linestyle="--", alpha=0.5)
    if i == 0:
        ax14[i].legend(loc="upper right")

ax14[2].set_xlabel("Tempo [Órbitas]")
fig14.suptitle("Velocidade Angular: Real vs. Medida pelo Giroscópio (MPU9250)", fontsize=12)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 15: Erro de Medição do Giroscópio (IMU MPU9250)
# ------------------------------------------------------------------------------
fig15, ax15 = plt.subplots(figsize=(9, 4))
ax15.plot(timeOrbits_imu, gyro_error_deg[:, 0], label=r'Erro $\omega_x$', color='tab:red', alpha=0.8, linewidth=1.0)
ax15.plot(timeOrbits_imu, gyro_error_deg[:, 1], label=r'Erro $\omega_y$', color='tab:green', alpha=0.8, linewidth=1.0)
ax15.plot(timeOrbits_imu, gyro_error_deg[:, 2], label=r'Erro $\omega_z$', color='tab:blue', alpha=0.8, linewidth=1.0)
ax15.set_title("Ruído e Drift do Giroscópio (MPU9250)")
ax15.set_xlabel("Tempo [Órbitas]")
ax15.set_ylabel("Erro [deg/s]")
ax15.legend(loc="upper right")
ax15.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 16: Comparação Aceleração Linear Real vs. Medida pelo Acelerômetro (MPU9250)
# ------------------------------------------------------------------------------
fig16, ax16 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
labels_a = [r'$a_x$ (Eixo $x$)', r'$a_y$ (Eixo $y$)', r'$a_z$ (Eixo $z$)']
colors_a = ['tab:red', 'tab:green', 'tab:blue']

for i in range(3):
    ax16[i].plot(timeOrbits, accel_real_g[:, i], label=r'Real ($F_{\text{ext}}/m$)', color='black', alpha=0.7, linewidth=1.5)
    ax16[i].plot(timeOrbits_imu, accel_measured_g[:, i], label=f'Medido IMU ({labels_a[i]})', color=colors_a[i], linestyle='--', alpha=0.8, linewidth=1.0)
    ax16[i].set_ylabel(f"{labels_a[i]}\n[g]")
    ax16[i].grid(True, linestyle="--", alpha=0.5)
    if i == 0:
        ax16[i].legend(loc="upper right")

ax16[2].set_xlabel("Tempo [Órbitas]")
fig16.suptitle("Aceleração Linear: Real vs. Medida pelo Acelerômetro (MPU9250)", fontsize=12)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 17: Erro de Medição do Acelerômetro (IMU MPU9250)
# ------------------------------------------------------------------------------
fig17, ax17 = plt.subplots(figsize=(9, 4))
ax17.plot(timeOrbits_imu, accel_error_mg[:, 0], label=r'Erro $a_x$', color='tab:red', alpha=0.8, linewidth=1.0)
ax17.plot(timeOrbits_imu, accel_error_mg[:, 1], label=r'Erro $a_y$', color='tab:green', alpha=0.8, linewidth=1.0)
ax17.plot(timeOrbits_imu, accel_error_mg[:, 2], label=r'Erro $a_z$', color='tab:blue', alpha=0.8, linewidth=1.0)
ax17.set_title("Ruído e Bias do Acelerômetro (MPU9250)")
ax17.set_xlabel("Tempo [Órbitas]")
ax17.set_ylabel("Erro [mg]")
ax17.legend(loc="upper right")
ax17.grid(True, linestyle="--", alpha=0.6)


plt.tight_layout()
plt.show()
